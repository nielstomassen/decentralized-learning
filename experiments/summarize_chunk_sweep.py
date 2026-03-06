#!/usr/bin/env python3
"""
Summarize chunk sweep: utility (test acc) and AUC per chunks_per_neighbor.

Reads MIA result CSVs from run_chunk_sweep.sh, groups by chunks_per_neighbor,
and reports mean (±std) of final-round global_test_acc and avg_auc so you can
pick an optimal chunks_per_neighbor (e.g. if 10 gives best AUC–utility tradeoff,
use --chunks-per-neighbor 10 or set topology so d≈10 and cpn=1).

Usage:
  python experiments/summarize_chunk_sweep.py --results-dir results/chunk_sweep
  python experiments/summarize_chunk_sweep.py --results-dir results/chunk_sweep --out-csv summary.csv
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd

CPN_REGEX = re.compile(r"_cpn([0-9]+)(?:_|\.)", re.IGNORECASE)


def parse_cpn_from_path(path: str) -> int | None:
    """Extract chunks_per_neighbor from filename (e.g. ..._cpn5_... or ..._cpn10_seed42.csv)."""
    base = os.path.basename(path)
    m = CPN_REGEX.search(base)
    return int(m.group(1)) if m else None


def load_and_label(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["round", "node_id", "global_test_acc", "avg_auc", "max_auc", "chunks_per_neighbor"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "chunks_per_neighbor" not in df.columns or df["chunks_per_neighbor"].isna().all():
        cpn = parse_cpn_from_path(path)
        if cpn is not None:
            df["chunks_per_neighbor"] = cpn
    df["_source_file"] = path
    return df


def main():
    p = argparse.ArgumentParser(description="Summarize chunk sweep: utility and AUC per chunks_per_neighbor.")
    p.add_argument("--results-dir", type=str, default="results/chunk_sweep", help="Directory containing CSV files.")
    p.add_argument("--results-glob", type=str, default=None, help="Glob for CSV paths (overrides results-dir).")
    p.add_argument("--round", type=int, default=None, help="Round to use (default: last round in each CSV).")
    p.add_argument("--by", type=str, choices=["chunks_per_neighbor", "degree"], default="chunks_per_neighbor",
                    help="Group by chunks_per_neighbor (cpn sweep) or degree (number-of-chunks sweep).")
    p.add_argument("--out-csv", type=str, default=None, help="If set, write summary table to this CSV.")
    args = p.parse_args()

    if args.results_glob:
        import glob
        paths = sorted(glob.glob(args.results_glob))
    else:
        paths = sorted(Path(args.results_dir).rglob("*.csv"))

    if not paths:
        print(f"No CSV files found under {args.results_dir}")
        return

    frames = []
    for path in paths:
        path = str(path)
        try:
            df = load_and_label(path)
            if args.by == "chunks_per_neighbor" and ("chunks_per_neighbor" not in df.columns or df["chunks_per_neighbor"].isna().all()):
                continue
            if args.by == "degree" and "degree" not in df.columns:
                continue
            frames.append(df)
        except Exception as e:
            print(f"Skip {path}: {e}")

    if not frames:
        print(f"No CSVs with required column ({args.by}) found.")
        return

    full = pd.concat(frames, ignore_index=True)

    # Use last round if not specified
    if args.round is not None:
        last_round = args.round
        full = full[full["round"] == last_round]
    else:
        last_round = int(full["round"].max())
        full = full[full["round"] == last_round]

    if full.empty:
        print("No rows for the selected round.")
        return

    group_col = args.by
    # Aggregate per group: mean and std over (seed, node_id)
    agg = full.groupby(group_col).agg(
        test_acc_mean=("global_test_acc", "mean"),
        test_acc_std=("global_test_acc", "std"),
        avg_auc_mean=("avg_auc", "mean"),
        avg_auc_std=("avg_auc", "std"),
        max_auc_mean=("max_auc", "mean"),
        max_auc_std=("max_auc", "std"),
        n=("node_id", "count"),
    ).reset_index()

    agg = agg.sort_values(group_col)

    print(f"Round: {last_round}  (use --round to override)  Group by: {group_col}")
    print()
    print(agg.to_string(index=False))
    print()
    if group_col == "chunks_per_neighbor":
        print("Pick chunks_per_neighbor with best tradeoff (high test_acc_mean, low avg_auc_mean = good utility, low leakage).")
    else:
        print("Pick degree (number of chunks) with best tradeoff; then use that topology (e.g. --regular-degree N) or ER p that gives ~N neighbors.")

    if args.out_csv:
        agg.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
