"""
One-time preparation:
  1. Download all Minari datasets used by tables 4 and 5.
  2. Build mujoco_norm.json: expert-relative normalization references for
     halfcheetah/hopper/walker2d (ref_max = mean return of the expert dataset's
     episodes; ref_min = mean return of a random policy over the recovered env).

Run once on a machine with internet (login node is fine; CPU is enough).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import minari
import gymnasium as gym

try:
    import gymnasium_robotics  # noqa: F401
    gym.register_envs(gymnasium_robotics)
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from datasets import ALL_DATASETS, MUJOCO_NORM_ENVS  # noqa: E402

N_RANDOM_EPISODES = 50
RANDOM_SEED = 0


def download_all():
    for ds in ALL_DATASETS:
        print(f"downloading {ds} ...", flush=True)
        minari.download_dataset(ds)


def expert_mean_return(env_name):
    ds = minari.load_dataset(f"mujoco/{env_name}/expert-v0", download=True)
    rets = [float(np.asarray(ep.rewards).sum()) for ep in ds.iterate_episodes()]
    return float(np.mean(rets))


def random_policy_return(env_name):
    ds = minari.load_dataset(f"mujoco/{env_name}/expert-v0", download=True)
    env = ds.recover_environment(eval_env=True)
    rng = np.random.default_rng(RANDOM_SEED)
    rets = []
    for i in range(N_RANDOM_EPISODES):
        obs, _ = env.reset(seed=RANDOM_SEED + i)
        done = False
        total = 0.0
        while not done:
            a = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(a)
            total += r
            done = bool(term or trunc)
        rets.append(total)
    env.close()
    return float(np.mean(rets))


def build_norm():
    norm = {}
    for env_name in MUJOCO_NORM_ENVS:
        rmax = expert_mean_return(env_name)
        rmin = random_policy_return(env_name)
        norm[env_name] = {"ref_min_score": rmin, "ref_max_score": rmax,
                          "convention": "expert-relative (expert-dataset mean=100, random policy=0)",
                          "n_random_episodes": N_RANDOM_EPISODES}
        print(f"{env_name}: ref_min(random)={rmin:.2f}  ref_max(expert)={rmax:.2f}", flush=True)
    out = os.path.join(HERE, "mujoco_norm.json")
    with open(out, "w") as f:
        json.dump(norm, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "download"):
        download_all()
    if what in ("all", "norm"):
        build_norm()
