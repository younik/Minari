"""Dataset / algorithm matrix for paper tables 4 (D4RL) and 5 (MuJoCo)."""

ALGOS = ["bc", "bc10", "td3_bc", "awac", "cql", "iql"]

# Table 4: D4RL navigation + manipulation (all on the Minari remote, carry ref scores).
D4RL_DATASETS = [
    "D4RL/pointmaze/umaze-v2",
    "D4RL/pointmaze/medium-v2",
    "D4RL/pointmaze/large-v2",
    "D4RL/antmaze/umaze-v1",
    "D4RL/antmaze/umaze-diverse-v1",
    "D4RL/antmaze/medium-play-v1",
    "D4RL/antmaze/medium-diverse-v1",
    "D4RL/antmaze/large-play-v1",
    "D4RL/antmaze/large-diverse-v1",
    "D4RL/pen/human-v2",
    "D4RL/pen/cloned-v2",
    "D4RL/pen/expert-v2",
    "D4RL/door/human-v2",
    "D4RL/door/cloned-v2",
    "D4RL/door/expert-v2",
    "D4RL/hammer/human-v2",
    "D4RL/hammer/cloned-v2",
    "D4RL/hammer/expert-v2",
    "D4RL/relocate/human-v2",
    "D4RL/relocate/cloned-v2",
    "D4RL/relocate/expert-v2",
]

# Table 5: MuJoCo locomotion. Only simple/medium/expert exist on the remote
# (simple-replay / medium-expert were never uploaded), so the table is built
# from these three variants per the agreed convention.
MUJOCO_DATASETS = [
    "mujoco/halfcheetah/simple-v0",
    "mujoco/halfcheetah/medium-v0",
    "mujoco/halfcheetah/expert-v0",
    "mujoco/hopper/simple-v0",
    "mujoco/hopper/medium-v0",
    "mujoco/hopper/expert-v0",
    "mujoco/walker2d/simple-v0",
    "mujoco/walker2d/medium-v0",
    "mujoco/walker2d/expert-v0",
]

# Envs needing an expert-relative normalization reference (no ref scores in metadata).
MUJOCO_NORM_ENVS = ["halfcheetah", "hopper", "walker2d"]

ALL_DATASETS = D4RL_DATASETS + MUJOCO_DATASETS


def all_jobs(seeds):
    jobs = []
    for ds in ALL_DATASETS:
        for algo in ALGOS:
            for seed in seeds:
                jobs.append((algo, ds, seed))
    return jobs
