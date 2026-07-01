"""
Collect results/*.json into the two paper tables (mean +/- std over seeds) and
emit LaTeX bodies matching tables 4 (D4RL) and 5 (MuJoCo) of paper/main.tex.

Usage:
    python collect.py            # prints a status summary + LaTeX
    python collect.py --score best   # use best-eval score instead of final
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)
from datasets import ALGOS, D4RL_DATASETS, MUJOCO_DATASETS  # noqa: E402

ALGO_HDR = {"bc": "BC", "bc10": "BC-10\\%", "td3_bc": "TD3+BC",
            "awac": "AWAC", "cql": "CQL", "iql": "IQL"}


def row_label(ds):
    # 'D4RL/pointmaze/umaze-v2' -> 'pointmaze/umaze-v2'; 'mujoco/halfcheetah/simple-v0' -> 'halfcheetah/simple-v0'
    return ds.split("/", 1)[1]


def _run_score(d, score_key, lastk):
    """Per-run scalar score. 'final' = last eval; 'best' = best eval over training;
    'lastk' = mean over the last K evaluations (more stable for noisy domains like
    antmaze, whose per-eval score oscillates a lot)."""
    if score_key == "best":
        return d["best_normalized_score"]
    if score_key == "lastk":
        evs = d.get("evaluations") or []
        norms = [e[1] for e in evs][-lastk:]
        return float(np.mean(norms)) if norms else d["normalized_score"]
    return d["normalized_score"]


def load_results(score_key, lastk=10):
    # results[dataset][algo] = list of per-seed scores
    res = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(HERE, "results", "*.json")):
        with open(path) as f:
            d = json.load(f)
        res[d["dataset"]][d["algo"]].append(_run_score(d, score_key, lastk))
    return res


def cell(scores):
    if not scores:
        return ""
    return f"{np.mean(scores):.1f}"


def group_of(ds):
    # 'D4RL/pointmaze/umaze-v2' -> 'pointmaze'; 'mujoco/halfcheetah/simple-v0' -> 'halfcheetah'
    return ds.split("/")[1]


def avg_cell(datasets, res, algo):
    """Average over the per-dataset mean scores for `algo` across `datasets`
    (only datasets that have at least one seed for this algo)."""
    means = [np.mean(res[ds][algo]) for ds in datasets
             if res.get(ds, {}).get(algo)]
    if not means:
        return ""
    return f"{np.mean(means):.1f}"


def emit_table(datasets, res, avg_after):
    """avg_after: dict {group_name: list_of_datasets_in_group} -> emit an
    '<group> average' row (and a \\midrule) after the last dataset of each group."""
    lines = []
    header = "                             & " + " & ".join(ALGO_HDR[a] for a in ALGOS) + r" \\ \midrule"
    lines.append(header)
    # which dataset is the last of each group (to place the average row after it)
    last_of = {dss[-1]: g for g, dss in avg_after.items()}
    for ds in datasets:
        cells = [cell(res.get(ds, {}).get(a, [])) for a in ALGOS]
        lines.append(f"{row_label(ds):<28} & " + " & ".join(f"{c:<5}" for c in cells) + r" \\")
        if ds in last_of:
            g = last_of[ds]
            acells = [avg_cell(avg_after[g], res, a) for a in ALGOS]
            lines.append(r"\midrule")
            lines.append(f"{g + ' average':<28} & " + " & ".join(f"{c:<5}" for c in acells) + r" \\")
            lines.append(r"\midrule")
    return "\n".join(lines)


def groups_in_order(datasets):
    """Ordered {group: [datasets...]} preserving dataset order."""
    out = {}
    for ds in datasets:
        out.setdefault(group_of(ds), []).append(ds)
    return out


def emit_detail(datasets, res):
    """mean +/- std per (dataset, algo), for the appendix / sanity-checking."""
    lines = []
    for ds in datasets:
        parts = []
        for a in ALGOS:
            s = res.get(ds, {}).get(a, [])
            if s:
                parts.append(f"{a}={np.mean(s):.1f}+/-{np.std(s):.1f}(n{len(s)})")
            else:
                parts.append(f"{a}=--")
        lines.append(f"  {row_label(ds):<28} " + "  ".join(parts))
    return "\n".join(lines)


def status(datasets, res, seeds_expected=3):
    done = miss = 0
    missing = []
    for ds in datasets:
        for a in ALGOS:
            n = len(res.get(ds, {}).get(a, []))
            done += n
            if n < seeds_expected:
                miss += seeds_expected - n
                missing.append(f"{a} {ds} ({n}/{seeds_expected})")
    return done, miss, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", choices=["final", "best", "lastk"], default="final")
    ap.add_argument("--lastk", type=int, default=10,
                    help="number of trailing evals to average when --score lastk")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    res = load_results(args.score, args.lastk)

    # Average rows after the pointmaze and antmaze navigation groups (table 4);
    # per-env averages for the locomotion groups (table 5).
    d4_avg = groups_in_order(D4RL_DATASETS)
    mj_avg = groups_in_order(MUJOCO_DATASETS)

    print("=" * 70)
    print(f"Score: {args.score}-eval normalized score, mean over seeds")
    for name, dsets in [("Table 4 (D4RL)", D4RL_DATASETS), ("Table 5 (MuJoCo)", MUJOCO_DATASETS)]:
        done, miss, missing = status(dsets, res)
        print(f"\n{name}: {done} runs collected, {miss} (algo,dataset,seed) still missing")
        if args.verbose and missing:
            for m in missing:
                print("   MISSING:", m)

    if args.verbose:
        print("\n--- per-(dataset,algo) mean +/- std ---")
        print("Table 4:"); print(emit_detail(D4RL_DATASETS, res))
        print("Table 5:"); print(emit_detail(MUJOCO_DATASETS, res))

    print("\n" + "=" * 70)
    print("%% ----- Table 4 (D4RL) body -----")
    print(emit_table(D4RL_DATASETS, res, d4_avg))
    print("\n%% ----- Table 5 (MuJoCo) body -----")
    print(emit_table(MUJOCO_DATASETS, res, mj_avg))


if __name__ == "__main__":
    main()
