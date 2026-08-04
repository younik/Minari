# Diagnostic plan: does state-normalization fix CQL on AntMaze?

## Hypothesis

CQL collapses to ~0 normalized score on **every** reproduced AntMaze dataset
(Table 4), while IQL/TD3+BC/AWAC learn on the *same* data + harness. We have ruled
out the usual suspects:

- **Config is not the cause.** `corl_configs/cql/antmaze/*.yaml` are byte-for-byte
  identical to CORL upstream (`corl-team/CORL/configs/offline/cql/antmaze/`) modulo
  seed/group/project bookkeeping.
- **Reward is not double-scaled.** The harness bakes `r·10−5` into the data once
  (`apply_reward_transform`); `ContinuousCQL` is built with no `reward_scale` arg
  (`run_benchmark.py:334`).
- **The pipeline works.** IQL on `antmaze/umaze-v1` (identical data/eval-frame/
  observation-flattening) reaches max **94.8** / last **84.3**.

The one behavioural difference that singles out CQL: it is the **only** algorithm
that runs with **state normalization off**.

| algo | `normalize` (state) |
|---|---|
| BC, BC-10%, TD3+BC, IQL | `true` |
| AWAC | unset → defaults `true` |
| **CQL** | **`false`** |

With `normalize: false`, `run_benchmark.py:452` sets `state_mean=0, state_std=1`, so
CQL sees **raw, unnormalized** observations. This harness's AntMaze state is the
non-standard `concat(observation[27], achieved_goal[2], desired_goal[2])` (31-dim),
where `achieved_goal`/`desired_goal` are **raw world-coordinate xy positions**
(≈0–36 on the large maze). Feeding large-magnitude, unnormalized goal coordinates to
CQL's magnitude-sensitive conservative Q-regularization is a plausible route to the
flat-zero collapse. `normalize: false` is faithful to CORL, but CORL applies it to
D4RL's *native* antmaze observations — not to this re-assembled layout.

**Prediction:** turning state normalization **on** for CQL on AntMaze lifts the score
off zero.

---

## Experiment

This is a **diagnostic** run, not a change to the reported table. The committed
CORL-faithful config (`normalize: false`) must stay untouched; we override at the
CLI and write results to a **separate** directory so the canonical Table-4 results in
`results/` are never overwritten.

### Step 0 — (optional, no training) confirm the mechanism is real

Before spending GPU time, dump the per-feature range of the flattened CQL AntMaze
state and confirm the appended goal dims dwarf the rest. If the goal coordinates are
*not* large relative to the body features, deprioritize this hypothesis.

```bash
conda activate minari
python - <<'PY'
import numpy as np, minari
from run_benchmark import flatten_obs_dict, load_and_flatten   # adjust to actual helpers
ds = minari.load_dataset("D4RL/antmaze/umaze-v1", download=True)
# build the same 31-dim flat state the harness feeds CQL, then:
# print per-dim min/max/std; expect dims 27..30 (goals) to have much larger range
PY
```

Expected: dims 27–30 (`achieved_goal`, `desired_goal`) show range/std an order of
magnitude above the body dims → consistent with the hypothesis.

### Step 1 — add a CLI override for state normalization

Add an opt-in flag to `run_benchmark.py` so the diagnostic is explicit and the
on-disk config is not edited.

In `main()` argparse block (near `run_benchmark.py:387`):

```python
ap.add_argument("--force_normalize", choices=["auto", "on", "off"], default="auto",
                help="Override config 'normalize' (state normalization). "
                     "auto = use config value (default).")
```

Where `normalize_states` is read (`run_benchmark.py:404`):

```python
normalize_states = cfg.get("normalize", True)
if args.force_normalize != "auto":
    normalize_states = (args.force_normalize == "on")
```

`auto` reproduces current behaviour exactly, so no existing result changes.

### Step 2 — the confirmatory single job (cheapest signal)

Run **one** job: CQL, `antmaze/umaze-v1`, seed 0, state normalization forced **on**.
Write to a separate file so canonical results are untouched.

```bash
mkdir -p results_normtest
python run_benchmark.py \
    --algo cql --dataset D4RL/antmaze/umaze-v1 --seed 0 \
    --max_timesteps 1000000 --force_normalize on \
    --out results_normtest/cql__D4RL_antmaze_umaze-v1__seed0__normON.json
```

**Baseline for comparison** (already on disk):
`results/cql__D4RL_antmaze_umaze-v1__seed0.json` → last 0.0, best 2.2, max-over-run 2.2.

**Decision gate:**
- **Score climbs off zero** (best or last ≫ 2, ideally approaching CORL's ~93) →
  hypothesis **confirmed**. Proceed to Step 3.
- **Still ~0** → hypothesis **rejected**; state normalization is not the cause.
  Stop; move to the data-semantics suspect (reward sparsity / `terminal` flags of the
  regenerated antmaze-v1 vs D4RL v2) — see "If rejected" below.

### Step 3 — full CQL × AntMaze grid (only if Step 2 is positive)

Re-run all **18** CQL AntMaze jobs (6 datasets × 3 seeds) with `--force_normalize on`,
each to `results_normtest/…__normON.json`. Exactly these jobs (they are the CQL
AntMaze lines of `jobs.txt`):

```
cql D4RL/antmaze/umaze-v1          0   1   2
cql D4RL/antmaze/umaze-diverse-v1  0   1   2
cql D4RL/antmaze/medium-play-v1    0   1   2
cql D4RL/antmaze/medium-diverse-v1 0   1   2
cql D4RL/antmaze/large-play-v1     0   1   2
cql D4RL/antmaze/large-diverse-v1  0   1   2
```

> Note: in the current `results/`, `cql large-diverse-v1` has only seeds 0–1 (n=2).
> Run all three seeds here so the grid is complete.

SLURM: copy the 18 lines into `jobs_normtest.txt`, add `--force_normalize on` and the
`results_normtest/` output path to the `srun` line in `submit_array.sh`, set
`--array=1-18`, and submit. Locally: a loop over the 18 `(dataset, seed)` pairs.

### Success criteria (Step 3)

Compare `results_normtest/` (norm-on) against the canonical `results/` (norm-off) and
CORL's published CQL AntMaze last/best:

| dataset | norm-off last (now) | CORL last | CORL best |
|---|---|---|---|
| umaze | 0.0 | 92.8 | 94.8 |
| umaze-diverse | 3.3 | 37.2 | 53.8 |
| medium-play | 6.5 | 65.8 | 80.5 |
| medium-diverse | 0.0 | 67.2 | 71.0 |
| large-play | 3.6 | 20.8 | 34.8 |
| large-diverse | 0.0 | 20.5 | 36.2 |

- **Strong confirmation:** norm-on CQL lands in the CORL ballpark (umaze/medium ≈
  50–90; large ≈ 15–35).
- **Partial:** climbs well off zero but below CORL → normalization is *a* cause but
  the regenerated data / observation layout also contributes; run the layout arm
  (Step 4).

### Step 4 — isolate normalization vs layout (only if Step 3 is partial)

To separate the two halves of the hypothesis, add one more diagnostic arm on
`umaze-v1` seed 0:
- **Layout arm:** keep `normalize: false` but reorder the flat state to put xy at the
  front (D4RL native order) — tests whether the *layout*, not normalization per se,
  is what CORL's `normalize: false` assumes.
- If the layout arm alone (no normalization) also recovers CQL, the true fix is to
  match D4RL's observation layout rather than to force normalization.

---

## If rejected (Step 2 stays ~0)

State normalization is not the cause. Next suspect: **regenerated-data semantics.**
Dump and compare, for `antmaze/umaze-v1`, the loaded `rewards` and `terminals`
distributions against D4RL antmaze-v2 (expect sparse 1-at-goal reward and
terminal-at-goal). CQL with `cql_max_target_backup: true` is uniquely sensitive to
`done`/timeout bootstrapping; a mismatch there would corrupt the `r·10−5` target for
CQL specifically while IQL tolerates it.

---

## Deliverable / how this feeds the paper

- **If confirmed:** the CQL-AntMaze zeros are a harness bug (state normalization off ×
  non-standard observation layout), not intrinsic CQL behaviour. Decide the fix —
  either normalize states for CQL on AntMaze in this harness (a documented, minimal
  deviation from CORL), or fix the observation layout so CORL's `normalize: false`
  behaves as upstream — regenerate the CQL AntMaze rows of Table 4, and correct the
  caption (which currently attributes the gap to reward-scaling/HP sensitivity).
- **If rejected:** update the Table 4 caption to state honestly that CQL fails to
  reproduce on the regenerated AntMaze despite a CORL-identical config and correct
  reward scaling, and pursue the data-semantics investigation.

## Guardrails

- Never write diagnostic runs into `results/`; use `results_normtest/`.
- Do **not** commit any change to `corl_configs/cql/antmaze/*.yaml` — the override is
  CLI-only so the reported table stays CORL-faithful.
- `--force_normalize auto` must be the default and must reproduce current numbers
  bit-for-bit (regression check: one `auto` run should match an existing `results/`
  file).
