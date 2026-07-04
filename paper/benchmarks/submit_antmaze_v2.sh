#!/bin/bash
#SBATCH --job-name=antmaze-v2
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-12:00:00
#SBATCH --array=1-108%50
#SBATCH --output=/network/scratch/o/omar.younis/Minari/paper/benchmarks/slurm_logs/%A_%a.out
#SBATCH --error=/network/scratch/o/omar.younis/Minari/paper/benchmarks/slurm_logs/%A_%a.out

set -u
BASE=/network/scratch/o/omar.younis
BENCH=$BASE/Minari/paper/benchmarks

# Keep every cache/dataset on scratch (home is over quota).
export MINARI_DATASETS_PATH=$BASE/minari_data
export MUJOCO_GL=egl
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME=$BASE/.hf_home
export XDG_CACHE_HOME=$BASE/.cache
export MPLCONFIGDIR=$BASE/.cache/mpl
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
mkdir -p "$BENCH/results" "$BENCH/slurm_logs" "$HF_HOME" "$XDG_CACHE_HOME"

PY=$BASE/corl_env/bin/python
JOBS=$BENCH/jobs_antmaze_v2.txt

# Ensure torch's bundled CUDA/cuDNN libs are found on every node.
SP=$BASE/corl_env/lib/python3.11/site-packages
NVLIB=$(echo $SP/nvidia/*/lib | tr ' ' ':')
export LD_LIBRARY_PATH="${NVLIB}:${SP}/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

read -r ALGO DATASET SEED < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS")
TAG="${ALGO}__$(echo "$DATASET" | tr '/' '_')__seed${SEED}"
OUT="$BENCH/results/${TAG}.json"

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID host=$(hostname) algo=$ALGO dataset=$DATASET seed=$SEED"
nvidia-smi -L || true

# Resume: skip jobs that already produced a complete result.
if [ -f "$OUT" ] && grep -q '"normalized_score"' "$OUT"; then
    echo "Already complete: $OUT  -- skipping."
    exit 0
fi

# Retry to absorb transient, node-specific import flakiness on the shared FS.
RC=1
for attempt in 1 2 3; do
    srun $PY "$BENCH/run_benchmark.py" \
        --algo "$ALGO" --dataset "$DATASET" --seed "$SEED" \
        --max_timesteps 1000000 --out "$OUT"
    RC=$?
    if [ "$RC" -eq 0 ] && [ -f "$OUT" ]; then break; fi
    echo "[$(date)] attempt $attempt failed (rc=$RC); retrying after backoff..." >&2
    sleep 15
done
echo "[$(date)] task=$SLURM_ARRAY_TASK_ID exit=$RC"
if [ "$RC" -eq 0 ] && [ ! -f "$OUT" ]; then
    echo "ERROR: exit 0 but no result file written" >&2
    RC=1
fi
exit $RC
