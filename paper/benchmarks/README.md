# CORL-on-Minari benchmarks (paper tables 4 & 5)

Reproduces the normalized-score tables in `paper/main.tex`
(`tab:benchmarks_d4rl` and `tab:benchmarks_mujoco`) by running the
[CORL](https://github.com/tinkoff-ai/CORL) offline-RL implementations on the
Minari datasets.

Six algorithms: **BC, BC-10%, TD3+BC, AWAC, CQL, IQL**.

## How it works

- `corl_vendor/` — the CORL single-file implementations, vendored verbatim with
  only their `d4rl/gym/wandb/pyrallis` coupling and `train()`/`__main__` stripped.
  The networks and trainer classes are used unchanged.
- `corl_configs/` — CORL's per-(algo, env) tuned hyperparameter YAMLs. Each Minari
  dataset is mapped to the closest config (`pointmaze→maze2d`, `antmaze→antmaze`,
  adroit→its own, mujoco `simple→medium`).
- `run_benchmark.py` — the harness: loads a Minari dataset, flattens goal-dict
  observations (see **AntMaze gotcha** below), recovers the eval env, applies the
  per-algo state/reward normalization, trains for 1e6 gradient steps, evaluates,
  and writes one JSON to `results/`.
- `datasets.py` — the dataset × algorithm matrix.
- `submit_array.sh` — SLURM array (one task per (algo, dataset, seed)).
- `collect.py` — aggregates `results/*.json` into the two LaTeX table bodies.

## Normalization

- **Table 4 (D4RL):** datasets carry `ref_min_score`/`ref_max_score`; normalized
  score = `100·(raw − ref_min)/(ref_max − ref_min)`.
- **Table 5 (MuJoCo):** these datasets have no reference scores, so we use an
  **expert-relative** convention (`mujoco_norm.json`): `ref_max` = mean return of
  the env's Minari *expert* dataset, `ref_min` = mean return of a random policy
  (50 episodes). `0` = random, `100` = Minari expert dataset.

## AntMaze gotcha (two fixes, both required)

Naively, antmaze produces **exactly 0.0** for every algorithm. Two distinct bugs:

1. **Observation flattening.** AntMaze's `observation` (27-dim = `qpos[2:]+qvel`)
   *excludes* the ant's global xy; that position lives in `achieved_goal`. The
   goal-conditioned policy needs its own position to navigate, so the flat state
   is `concat(observation, achieved_goal, desired_goal)` (31-dim). Using only
   `concat(observation, desired_goal)` tells the ant where the goal is but not
   where *it* is, so it never moves (eval displacement ~0.1) and scores 0.
   PointMaze's `observation` already contains position, so it keeps
   `concat(observation, desired_goal)`.
2. **Eval coordinate frame.** The dataset's stored `eval_env_spec` uses a maze_map
   that is rotated 180° relative to the data-generation `env_spec` (a negated
   cell→world frame). The dataset observations live in the `env_spec` frame, so we
   evaluate antmaze with `recover_environment(eval_env=False)` to match. Other
   groups have a consistent `eval_env_spec` and use `eval_env=True`.

With both fixes, td3_bc on antmaze/umaze climbs from 0 to ~50 normalized by 150k
gradient steps (the ant reaches the goal). Diagnostics: `diag_antmaze*.py`,
`diag_dynamics.py`, `diag_rollout.py`.

## Caveats / decisions

- **3 seeds (0,1,2), 1e6 gradient steps**, matching CORL's tuned hyperparameters.
- **Table 5 variants:** the remote only ships `simple`/`medium`/`expert` MuJoCo
  variants. The original table's `simple-replay` and `medium-expert` were never
  uploaded, so the table is built from `simple`/`medium`/`expert` for
  halfcheetah/hopper/walker2d.

## Run

```bash
BASE=/network/scratch/o/omar.younis
export MINARI_DATASETS_PATH=$BASE/minari_data
# one-time: download datasets + build mujoco normalization refs
$BASE/corl_env/bin/python prep.py all
# submit all 540 jobs (already done: array 9988103)
sbatch submit_array.sh
# monitor
squeue -u $USER -n corl-minari
# build tables once results are in (resume-safe; re-run sbatch to fill gaps)
$BASE/corl_env/bin/python collect.py --score final --verbose
```

Environment: `/network/scratch/o/omar.younis/corl_env` (Python 3.11 venv).
Everything lives on scratch because `$HOME` is over its 100 GB quota.
