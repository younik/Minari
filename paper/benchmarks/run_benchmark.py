"""
Run a CORL offline-RL algorithm on a Minari dataset and report the normalized score.

This adapts the single-file reference implementations from CORL
(https://github.com/tinkoff-ai/CORL) to load Minari datasets instead of D4RL.
The algorithm networks and training code are vendored verbatim in ``corl_vendor/``;
only the data-loading, environment-recovery, normalization and train/eval loop
are reimplemented here on top of Minari + Gymnasium.

Algorithms: bc, bc10, td3_bc, awac, cql, iql.

Usage:
    python run_benchmark.py --algo iql --dataset D4RL/pen/human-v2 --seed 0 \
        --max_timesteps 1000000 --out results/iql__D4RL_pen_human-v2__0.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import yaml

import minari
import gymnasium as gym

# Registers Adroit (pen/door/hammer/relocate) and Maze (pointmaze/antmaze) envs.
try:
    import gymnasium_robotics  # noqa: F401
    gym.register_envs(gymnasium_robotics)
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from corl_vendor import bc as bc_mod  # noqa: E402
from corl_vendor import td3_bc as td3_mod  # noqa: E402
from corl_vendor import awac as awac_mod  # noqa: E402
from corl_vendor import cql as cql_mod  # noqa: E402
from corl_vendor import iql as iql_mod  # noqa: E402

LOCO = ("halfcheetah", "hopper", "walker2d")


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def group_of(dataset_id: str) -> str:
    """e.g. 'D4RL/antmaze/umaze-v1' -> 'antmaze', 'mujoco/halfcheetah/medium-v0' -> 'halfcheetah'."""
    parts = dataset_id.split("/")
    return parts[1] if len(parts) >= 3 else parts[0]


def dict_obs_keys(obs_space, group):
    """Order of goal-dict keys to concatenate into the flat policy state.

    AntMaze's ``observation`` (qpos[2:] + qvel) excludes the ant's global xy,
    which instead lives in ``achieved_goal``. The goal-conditioned policy needs
    its own position to navigate, so we must include ``achieved_goal``; without
    it the ant cannot tell which way the goal is and scores exactly 0 for every
    algorithm. PointMaze's ``observation`` already contains the ball's position,
    so ``achieved_goal`` is redundant there and we keep (observation, desired_goal).
    """
    order = (("observation", "achieved_goal", "desired_goal") if group == "antmaze"
             else ("observation", "desired_goal"))
    return [k for k in order if k in obs_space.spaces]


def make_flattener(obs_space, group) -> Tuple[Callable, int]:
    """Return (flatten_fn, state_dim). Dict (goal) spaces -> concat of dict_obs_keys."""
    if isinstance(obs_space, gym.spaces.Dict):
        keys = dict_obs_keys(obs_space, group)

        def flat(obs):
            return np.concatenate(
                [np.asarray(obs[k], dtype=np.float32).ravel() for k in keys]
            )

        dim = sum(int(np.prod(obs_space.spaces[k].shape)) for k in keys)
        return flat, dim
    else:
        def flat(obs):
            return np.asarray(obs, dtype=np.float32).ravel()

        return flat, int(np.prod(obs_space.shape))


def _flatten_episode_obs(obs, keys):
    """Vectorized flatten of a full episode's observations (length T+1)."""
    if isinstance(obs, dict):
        return np.concatenate(
            [np.asarray(obs[k], dtype=np.float32).reshape(len(obs[k]), -1) for k in keys],
            axis=1,
        )
    return np.asarray(obs, dtype=np.float32)


def load_dataset_arrays(ds, obs_space, group):
    """Convert a Minari dataset into the CORL/d4rl transition dict + per-episode returns."""
    keys = None
    if isinstance(obs_space, gym.spaces.Dict):
        keys = dict_obs_keys(obs_space, group)

    obs_l, next_l, act_l, rew_l, term_l = [], [], [], [], []
    ep_returns = []          # raw (undiscounted) episode returns
    ep_slices = []           # (start, end) indices into the flat arrays, per episode
    cursor = 0
    for ep in ds.iterate_episodes():
        flat = _flatten_episode_obs(ep.observations, keys)  # (T+1, D)
        a = np.asarray(ep.actions, dtype=np.float32)
        r = np.asarray(ep.rewards, dtype=np.float32)
        term = np.asarray(ep.terminations, dtype=np.float32)
        T = len(a)
        if T == 0:
            continue
        obs_l.append(flat[:-1])
        next_l.append(flat[1:])
        act_l.append(a)
        rew_l.append(r)
        term_l.append(term)
        ep_returns.append(float(r.sum()))
        ep_slices.append((cursor, cursor + T))
        cursor += T

    data = {
        "observations": np.concatenate(obs_l, axis=0),
        "next_observations": np.concatenate(next_l, axis=0),
        "actions": np.concatenate(act_l, axis=0),
        "rewards": np.concatenate(rew_l, axis=0),
        "terminals": np.concatenate(term_l, axis=0),
    }
    return data, np.asarray(ep_returns), ep_slices


def keep_best_fraction(data, ep_returns, ep_slices, frac, discount):
    """BC-10%: keep transitions from the top-`frac` episodes by discounted return."""
    disc_returns = []
    for (s, e) in ep_slices:
        r = data["rewards"][s:e]
        w = discount ** np.arange(len(r))
        disc_returns.append(float((w * r).sum()))
    order = np.argsort(disc_returns)[::-1]
    top = order[: max(1, int(frac * len(order)))]
    idx = np.concatenate([np.arange(*ep_slices[i]) for i in top])
    return {k: v[idx] for k, v in data.items()}


def apply_reward_transform(data, algo, group, normalize_reward,
                           reward_scale, reward_bias, ep_returns, max_ep_steps,
                           cql_reward_mode="corl"):
    """Replicate CORL's modify_reward (per-algo).

    ``cql_reward_mode`` is a diagnostic override for the CQL antmaze reward:
      - "corl"      : CORL-faithful ``r*reward_scale + reward_bias`` (default).
      - "minus_one" : ``r - 1`` (the same transform IQL/TD3+BC/AWAC use for
                      antmaze). CORL's scale/bias recipe assumes D4RL's *sparse*
                      reward; on Minari's dense continuing-task antmaze it becomes
                      a mostly-+5 dense signal. This arm tests whether the
                      sparse-tuned reward transform (not CQL itself) is the cause.
    """
    r = data["rewards"]
    if algo in ("iql", "td3_bc", "awac"):
        if normalize_reward:
            if group in LOCO:
                rng = ep_returns.max() - ep_returns.min()
                r = r / rng * max_ep_steps
            elif group == "antmaze":
                r = r - 1.0
    elif algo == "cql":
        if normalize_reward:
            if group in LOCO:
                rng = ep_returns.max() - ep_returns.min()
                r = r / rng * max_ep_steps
            if group == "antmaze" and cql_reward_mode == "minus_one":
                r = r - 1.0
            else:
                r = r * reward_scale + reward_bias
    data["rewards"] = r.astype(np.float32)
    return data


# --------------------------------------------------------------------------- #
# Normalization references
# --------------------------------------------------------------------------- #
def normalized_score(dataset_id, ds, raw_return, mujoco_ref):
    meta = ds.storage.metadata
    rmin = meta.get("ref_min_score")
    rmax = meta.get("ref_max_score")
    if rmin is not None and rmax is not None:
        return 100.0 * (raw_return - rmin) / (rmax - rmin)
    # MuJoCo: expert-relative normalization computed offline
    group = group_of(dataset_id)
    if group in mujoco_ref:
        rmin = mujoco_ref[group]["ref_min_score"]
        rmax = mujoco_ref[group]["ref_max_score"]
        return 100.0 * (raw_return - rmin) / (rmax - rmin)
    return raw_return  # fall back to raw if no reference available


# --------------------------------------------------------------------------- #
# Config (CORL tuned hyperparameters)
# --------------------------------------------------------------------------- #
ALGO_DIR = {"bc": "bc", "bc10": "bc_10", "td3_bc": "td3_bc",
            "awac": "awac", "cql": "cql", "iql": "iql"}


def config_path_for(algo, dataset_id):
    """Map a Minari dataset id to the closest CORL tuned-config yaml."""
    group = group_of(dataset_id)
    name = dataset_id.split("/")[-1]              # e.g. 'umaze-diverse-v1'
    base, _, _ = name.rpartition("-")             # strip version -> 'umaze-diverse'
    base = base.replace("-", "_")
    algo_dir = os.path.join(HERE, "corl_configs", ALGO_DIR[algo])

    if group == "pointmaze":
        cfgdir, fname = "maze2d", f"{base}_v1.yaml"
    elif group == "antmaze":
        cfgdir, fname = "antmaze", f"{base}_v2.yaml"
    elif group in ("pen", "door", "hammer", "relocate"):
        cfgdir, fname = group, f"{base}_v1.yaml"
    elif group in LOCO:
        # simple -> medium (closest analog); medium/expert direct
        m = {"simple": "medium", "medium": "medium", "expert": "expert"}.get(base, base)
        cfgdir, fname = group, f"{m}_v2.yaml"
    else:
        return None
    p = os.path.join(algo_dir, cfgdir, fname)
    return p if os.path.exists(p) else None


def _coerce_numbers(cfg):
    """YAML 1.1 parses '3e-4' (no decimal point) as a string; coerce such values to float."""
    for k, v in list(cfg.items()):
        if isinstance(v, str):
            s = v.strip()
            if s.lower() in ("inf", ".inf", "+inf"):
                cfg[k] = float("inf")
            elif s.lower() in ("-inf", "-.inf"):
                cfg[k] = float("-inf")
            else:
                try:
                    cfg[k] = float(s)
                except ValueError:
                    pass
    return cfg


def load_config(algo, dataset_id):
    cfg = {}
    p = config_path_for(algo, dataset_id)
    if p is not None:
        with open(p) as f:
            cfg = yaml.safe_load(f) or {}
    return _coerce_numbers(cfg)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(env, actor, n_episodes, flatten, state_mean, state_std, device, seed):
    actor.eval()
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + 1000 + i)
        done = False
        total = 0.0
        while not done:
            s = (flatten(obs) - state_mean) / state_std
            action = actor.act(s.astype(np.float32), device)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = bool(term or trunc)
        returns.append(total)
    actor.train()
    return np.asarray(returns)


# --------------------------------------------------------------------------- #
# Trainer construction (per algorithm), following CORL train() wiring
# --------------------------------------------------------------------------- #
def build_trainer(algo, cfg, state_dim, action_dim, max_action, max_steps, device):
    if algo in ("bc", "bc10"):
        actor = bc_mod.Actor(state_dim, action_dim, max_action).to(device)
        opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
        trainer = bc_mod.BC(max_action, actor, opt, discount=cfg.get("discount", 0.99), device=device)
        return trainer, actor

    if algo == "td3_bc":
        actor = td3_mod.Actor(state_dim, action_dim, max_action).to(device)
        a_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
        c1 = td3_mod.Critic(state_dim, action_dim).to(device)
        c1_opt = torch.optim.Adam(c1.parameters(), lr=3e-4)
        c2 = td3_mod.Critic(state_dim, action_dim).to(device)
        c2_opt = torch.optim.Adam(c2.parameters(), lr=3e-4)
        trainer = td3_mod.TD3_BC(
            max_action=max_action, actor=actor, actor_optimizer=a_opt,
            critic_1=c1, critic_1_optimizer=c1_opt, critic_2=c2, critic_2_optimizer=c2_opt,
            discount=cfg.get("discount", 0.99), tau=cfg.get("tau", 0.005),
            policy_noise=cfg.get("policy_noise", 0.2) * max_action,
            noise_clip=cfg.get("noise_clip", 0.5) * max_action,
            policy_freq=cfg.get("policy_freq", 2), alpha=cfg.get("alpha", 2.5), device=device)
        return trainer, actor

    if algo == "awac":
        hidden = cfg.get("hidden_dim", 256)
        actor = awac_mod.Actor(state_dim, action_dim, hidden,
                               min_action=-max_action, max_action=max_action).to(device)
        lr = cfg.get("learning_rate", 3e-4)
        a_opt = torch.optim.Adam(actor.parameters(), lr=lr)
        c1 = awac_mod.Critic(state_dim, action_dim, hidden).to(device)
        c2 = awac_mod.Critic(state_dim, action_dim, hidden).to(device)
        c1_opt = torch.optim.Adam(c1.parameters(), lr=lr)
        c2_opt = torch.optim.Adam(c2.parameters(), lr=lr)
        trainer = awac_mod.AdvantageWeightedActorCritic(
            actor=actor, actor_optimizer=a_opt,
            critic_1=c1, critic_1_optimizer=c1_opt, critic_2=c2, critic_2_optimizer=c2_opt,
            gamma=cfg.get("gamma", 0.99), tau=cfg.get("tau", 5e-3),
            awac_lambda=cfg.get("awac_lambda", 1.0))
        return trainer, actor

    if algo == "cql":
        orth = cfg.get("orthogonal_init", True)
        qn = cfg.get("q_n_hidden_layers", 3)
        c1 = cql_mod.FullyConnectedQFunction(state_dim, action_dim, orth, qn).to(device)
        c2 = cql_mod.FullyConnectedQFunction(state_dim, action_dim, orth, qn).to(device)
        c1_opt = torch.optim.Adam(list(c1.parameters()), cfg.get("qf_lr", 3e-4))
        c2_opt = torch.optim.Adam(list(c2.parameters()), cfg.get("qf_lr", 3e-4))
        actor = cql_mod.TanhGaussianPolicy(
            state_dim, action_dim, max_action,
            log_std_multiplier=cfg.get("policy_log_std_multiplier", 1.0),
            orthogonal_init=orth).to(device)
        a_opt = torch.optim.Adam(actor.parameters(), cfg.get("policy_lr", 3e-5))
        trainer = cql_mod.ContinuousCQL(
            critic_1=c1, critic_2=c2, critic_1_optimizer=c1_opt, critic_2_optimizer=c2_opt,
            actor=actor, actor_optimizer=a_opt, discount=cfg.get("discount", 0.99),
            soft_target_update_rate=cfg.get("soft_target_update_rate", 5e-3), device=device,
            target_entropy=-float(action_dim),
            alpha_multiplier=cfg.get("alpha_multiplier", 1.0),
            use_automatic_entropy_tuning=cfg.get("use_automatic_entropy_tuning", True),
            backup_entropy=cfg.get("backup_entropy", False),
            policy_lr=cfg.get("policy_lr", 3e-5), qf_lr=cfg.get("qf_lr", 3e-4),
            bc_steps=cfg.get("bc_steps", 0), target_update_period=cfg.get("target_update_period", 1),
            cql_n_actions=cfg.get("cql_n_actions", 10),
            cql_importance_sample=cfg.get("cql_importance_sample", True),
            cql_lagrange=cfg.get("cql_lagrange", False),
            cql_target_action_gap=cfg.get("cql_target_action_gap", -1.0),
            cql_temp=cfg.get("cql_temp", 1.0), cql_alpha=cfg.get("cql_alpha", 10.0),
            cql_max_target_backup=cfg.get("cql_max_target_backup", False),
            cql_clip_diff_min=cfg.get("cql_clip_diff_min", -np.inf),
            cql_clip_diff_max=cfg.get("cql_clip_diff_max", np.inf))
        return trainer, actor

    if algo == "iql":
        q = iql_mod.TwinQ(state_dim, action_dim).to(device)
        v = iql_mod.ValueFunction(state_dim).to(device)
        dropout = cfg.get("actor_dropout", None)
        if cfg.get("iql_deterministic", False):
            actor = iql_mod.DeterministicPolicy(state_dim, action_dim, max_action, dropout=dropout).to(device)
        else:
            actor = iql_mod.GaussianPolicy(state_dim, action_dim, max_action, dropout=dropout).to(device)
        v_opt = torch.optim.Adam(v.parameters(), lr=cfg.get("vf_lr", 3e-4))
        q_opt = torch.optim.Adam(q.parameters(), lr=cfg.get("qf_lr", 3e-4))
        a_opt = torch.optim.Adam(actor.parameters(), lr=cfg.get("actor_lr", 3e-4))
        trainer = iql_mod.ImplicitQLearning(
            max_action=max_action, actor=actor, actor_optimizer=a_opt,
            q_network=q, q_optimizer=q_opt, v_network=v, v_optimizer=v_opt,
            iql_tau=cfg.get("iql_tau", 0.7), beta=cfg.get("beta", 3.0),
            max_steps=max_steps, discount=cfg.get("discount", 0.99),
            tau=cfg.get("tau", 0.005), device=device)
        return trainer, actor

    raise ValueError(algo)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=list(ALGO_DIR.keys()))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_timesteps", type=int, default=1_000_000)
    ap.add_argument("--eval_freq", type=int, default=0, help="0 = use config default")
    ap.add_argument("--n_episodes", type=int, default=0, help="0 = use config default")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mujoco_ref", default=os.path.join(HERE, "mujoco_norm.json"))
    ap.add_argument("--force_normalize", choices=["auto", "on", "off"], default="auto",
                    help="Override config 'normalize' (state normalization). "
                         "auto = use config value (default; reproduces committed results).")
    ap.add_argument("--cql_reward_mode", choices=["corl", "minus_one"], default="corl",
                    help="Diagnostic: CQL antmaze reward transform. corl = CORL-faithful "
                         "r*scale+bias (default). minus_one = r-1 (IQL-style).")
    args = ap.parse_args()

    t0 = time.time()
    set_seed(args.seed)
    device = args.device

    mujoco_ref = {}
    if os.path.exists(args.mujoco_ref):
        with open(args.mujoco_ref) as f:
            mujoco_ref = json.load(f)

    cfg = load_config(args.algo, args.dataset)
    group = group_of(args.dataset)
    normalize_states = cfg.get("normalize", True)
    if args.force_normalize != "auto":
        normalize_states = (args.force_normalize == "on")
    normalize_reward = cfg.get("normalize_reward", False)
    reward_scale = cfg.get("reward_scale", 1.0)
    reward_bias = cfg.get("reward_bias", 0.0)
    eval_freq = args.eval_freq or cfg.get("eval_freq", 5000)
    n_episodes = args.n_episodes or cfg.get("n_episodes", 10)

    print(f"[{args.algo}] {args.dataset} seed={args.seed} | "
          f"normalize_states={normalize_states} normalize_reward={normalize_reward} "
          f"reward_scale={reward_scale} reward_bias={reward_bias} "
          f"cql_reward_mode={args.cql_reward_mode} "
          f"eval_freq={eval_freq} n_episodes={n_episodes}", flush=True)

    ds = minari.load_dataset(args.dataset, download=True)
    # AntMaze's stored eval_env_spec uses a maze_map that is rotated 180 deg
    # relative to the data-generation env_spec, i.e. a different cell->world
    # coordinate frame. The dataset's observations (achieved_goal/desired_goal)
    # live in the env_spec frame, so evaluating in the eval_env_spec frame feeds
    # the goal-conditioned policy out-of-distribution goals and yields exactly 0
    # reward for every algorithm. Recover the data-generation env for antmaze so
    # the eval frame matches the dataset. Other groups (pointmaze, adroit) have a
    # consistent eval_env_spec and keep using it.
    use_eval_env_spec = group_of(args.dataset) != "antmaze"
    eval_env = ds.recover_environment(eval_env=use_eval_env_spec)
    max_ep_steps = eval_env.spec.max_episode_steps or 1000
    obs_space = eval_env.observation_space
    flatten, state_dim = make_flattener(obs_space, group)
    action_dim = int(np.prod(eval_env.action_space.shape))
    max_action = float(eval_env.action_space.high[0])

    data, ep_returns, ep_slices = load_dataset_arrays(ds, obs_space, group)
    print(f"  transitions={len(data['rewards'])} episodes={len(ep_slices)} "
          f"state_dim={state_dim} action_dim={action_dim} max_action={max_action} "
          f"max_ep_steps={max_ep_steps}", flush=True)

    # BC-10% trajectory filtering
    if args.algo == "bc10":
        data = keep_best_fraction(data, ep_returns, ep_slices,
                                  frac=cfg.get("frac", 0.1), discount=cfg.get("discount", 0.99))
        print(f"  bc10 kept transitions={len(data['rewards'])}", flush=True)

    # Reward transform
    data = apply_reward_transform(data, args.algo, group, normalize_reward,
                                  reward_scale, reward_bias, ep_returns, max_ep_steps,
                                  cql_reward_mode=args.cql_reward_mode)

    # State normalization
    if normalize_states:
        state_mean, state_std = iql_mod.compute_mean_std(data["observations"], eps=1e-3)
    else:
        state_mean = np.zeros(state_dim, dtype=np.float32)
        state_std = np.ones(state_dim, dtype=np.float32)
    data["observations"] = (data["observations"] - state_mean) / state_std
    data["next_observations"] = (data["next_observations"] - state_mean) / state_std

    # Replay buffer (GPU tensors)
    n = len(data["rewards"])
    states = torch.tensor(data["observations"], dtype=torch.float32, device=device)
    actions = torch.tensor(data["actions"], dtype=torch.float32, device=device)
    rewards = torch.tensor(data["rewards"], dtype=torch.float32, device=device).unsqueeze(-1)
    next_states = torch.tensor(data["next_observations"], dtype=torch.float32, device=device)
    dones = torch.tensor(data["terminals"], dtype=torch.float32, device=device).unsqueeze(-1)

    def sample(bs):
        idx = torch.randint(0, n, (bs,), device=device)
        return [states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx]]

    trainer, actor = build_trainer(args.algo, cfg, state_dim, action_dim,
                                   max_action, args.max_timesteps, device)

    evaluations = []  # list of (step, normalized_mean, raw_mean)
    for t in range(args.max_timesteps):
        batch = sample(args.batch_size)
        if args.algo == "awac":
            trainer.update(batch)
        else:
            trainer.train(batch)
        if (t + 1) % eval_freq == 0 or (t + 1) == args.max_timesteps:
            raw = evaluate(eval_env, actor, n_episodes, flatten,
                           state_mean, state_std, device, args.seed)
            raw_mean = float(raw.mean())
            norm = normalized_score(args.dataset, ds, raw_mean, mujoco_ref)
            evaluations.append((t + 1, norm, raw_mean))
            print(f"  step {t+1}: raw={raw_mean:.2f}  normalized={norm:.2f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    last_norm = evaluations[-1][1]
    best_norm = max(e[1] for e in evaluations)
    result = {
        "algo": args.algo,
        "dataset": args.dataset,
        "seed": args.seed,
        "max_timesteps": args.max_timesteps,
        "normalized_score": last_norm,        # final-step score
        "best_normalized_score": best_norm,
        "raw_return": evaluations[-1][2],
        "n_episodes": n_episodes,
        "evaluations": evaluations,
        "wall_time_s": time.time() - t0,
        "config_path": config_path_for(args.algo, args.dataset),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"DONE -> {args.out}: final={last_norm:.2f} best={best_norm:.2f}", flush=True)


if __name__ == "__main__":
    main()
