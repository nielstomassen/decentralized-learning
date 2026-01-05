#!/usr/bin/env python3
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Examples your filenames might contain:
#   ..._alpha0.1_beta0.2_..._seed123.csv
#   ...{"topology_name":"regular","regular_degree":1}...
ALPHA_REGEX = re.compile(r"alpha([0-9]*\.?[0-9]+)", re.IGNORECASE)
SEED_REGEX = re.compile(r"seed(\d+)", re.IGNORECASE)

# Robust-ish "degree" extraction for filenames that embed JSON-ish snippets
# Matches: regular_degree": 1  OR  regular_degree:1  OR  regular_degree_1 etc.
DEGREE_REGEX = re.compile(r"regular_degree[^0-9]*([0-9]+)", re.IGNORECASE)


def parse_alpha_from_filename(path: str) -> float:
    m = ALPHA_REGEX.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Could not find 'alphaX' in filename: {path}")
    return float(m.group(1))


def parse_seed_from_filename(path: str) -> int | None:
    m = SEED_REGEX.search(os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))


def parse_degree_from_filename(path: str) -> int:
    m = DEGREE_REGEX.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Could not find 'regular_degree' in filename: {path}")
    return int(m.group(1))


def summarize_by_round_mean_over_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input CSV: per-node per-round rows with columns:
      round,node_id,train_acc,test_acc,auc

    Output: one row per round with mean train_acc/test_acc/auc over nodes.
    Empty strings become NaN and are ignored by mean().
    """
    required = {"round", "node_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Coerce numeric (early rounds often empty -> NaN)
    for c in ["round", "node_id", "train_acc", "test_acc", "auc"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    agg_cols = [c for c in ["train_acc", "test_acc", "auc"] if c in df.columns]
    out = (
        df.groupby("round", as_index=False)[agg_cols]
        .mean()
        .sort_values("round")
        .reset_index(drop=True)
    )
    return out


def pick_fixed_or_last_row(df_round: pd.DataFrame, fixed_round: int, metric_cols=("auc", "test_acc")) -> pd.Series:
    """
    From per-round aggregated df (mean-over-nodes):
      - drop rounds where all metric_cols are NaN
      - return row at fixed_round if available else last usable row
    """
    cols_present = [c for c in metric_cols if c in df_round.columns]
    if not cols_present:
        raise KeyError(f"None of the metric cols {metric_cols} present in df.")

    df2 = df_round.dropna(subset=cols_present, how="all").sort_values("round")
    if df2.empty:
        raise ValueError("No usable metric rows after dropping NaNs.")

    if fixed_round in df2["round"].values:
        return df2.loc[df2["round"] == fixed_round].iloc[0]
    return df2.iloc[-1]


def load_runs(files: list[str], fixed_round: int) -> list[dict]:
    """
    For each file:
      - parse alpha, degree, seed
      - read CSV, aggregate mean-over-nodes per round
      - pick auc/test_acc at fixed_round if present else last usable round
    Returns list of dict rows (one per file/run).
    """
    runs = []
    for f in files:
        alpha = parse_alpha_from_filename(f)
        degree = parse_degree_from_filename(f)
        seed = parse_seed_from_filename(f)

        raw = pd.read_csv(f)
        per_round = summarize_by_round_mean_over_nodes(raw)

        row = pick_fixed_or_last_row(per_round, fixed_round, metric_cols=("auc", "test_acc"))
        runs.append(
            {
                "alpha": alpha,
                "degree": degree,
                "seed": seed,
                "picked_round": int(row["round"]),
                "auc": float(row["auc"]) if "auc" in row.index and pd.notna(row["auc"]) else np.nan,
                "test_acc": float(row["test_acc"]) if "test_acc" in row.index and pd.notna(row["test_acc"]) else np.nan,
                "source_file": os.path.basename(f),
            }
        )
    return runs


def aggregate_over_seeds(runs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across runs (typically different seeds) for each (alpha, degree).
    Nodes are *already* averaged within a run when we computed per-round mean-over-nodes.

    Output columns:
      alpha, degree, n_runs, mean_auc, std_auc, mean_acc, std_acc
    """
    def _agg(group: pd.DataFrame) -> pd.Series:
        auc_vals = group["auc"].dropna().to_numpy(dtype=float)
        acc_vals = group["test_acc"].dropna().to_numpy(dtype=float)
        return pd.Series(
            {
                "n_runs_auc": int(auc_vals.size),
                "mean_auc": float(np.mean(auc_vals)) if auc_vals.size else np.nan,
                "std_auc": float(np.std(auc_vals, ddof=0)) if auc_vals.size else np.nan,
                "n_runs_acc": int(acc_vals.size),
                "mean_acc": float(np.mean(acc_vals)) if acc_vals.size else np.nan,
                "std_acc": float(np.std(acc_vals, ddof=0)) if acc_vals.size else np.nan,
            }
        )

    out = (
        runs_df.groupby(["alpha", "degree"], as_index=False)
        .apply(_agg, include_groups=False)
        .reset_index(drop=True)
        .sort_values(["alpha", "degree"])
        .reset_index(drop=True)
    )
    return out


def plot_auc_vs_degree_per_alpha(agg: pd.DataFrame, out_dir: str, fixed_round: int):
    """
    Make one plot per alpha:
      x = degree
      y = mean_auc (error bars = std_auc across seeds)
    """
    os.makedirs(out_dir, exist_ok=True)

    for alpha in sorted(agg["alpha"].unique()):
        df = agg[agg["alpha"] == alpha].dropna(subset=["degree", "mean_auc", "std_auc"]).sort_values("degree")
        if df.empty:
            print(f"[WARN] No plottable rows for alpha={alpha}. Skipping.")
            continue

        plt.figure()
        plt.errorbar(
            df["degree"].values,
            df["mean_auc"].values,
            yerr=df["std_auc"].values,
            fmt="o-",
            capsize=4,
        )
        plt.xlabel("Topology degree (regular_degree)")
        plt.ylabel("Mean AUC (mean over nodes per run, then mean over seeds)")
        plt.title(f"AUC vs topology degree @ round {fixed_round} (alpha={alpha})")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)

        out_path = os.path.join(out_dir, f"auc_vs_degree_alpha{alpha}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()


def plot_auc_vs_degree_all_alphas(agg: pd.DataFrame, out_dir: str, fixed_round: int):
    """
    Optional: one combined plot with multiple alpha curves.
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()

    for alpha in sorted(agg["alpha"].unique()):
        df = agg[agg["alpha"] == alpha].dropna(subset=["degree", "mean_auc", "std_auc"]).sort_values("degree")
        if df.empty:
            continue
        plt.errorbar(
            df["degree"].values,
            df["mean_auc"].values,
            yerr=df["std_auc"].values,
            fmt="o-",
            capsize=4,
            label=f"alpha={alpha}",
        )

    plt.xlabel("Topology degree (regular_degree)")
    plt.ylabel("Mean AUC (mean over nodes per run, then mean over seeds)")
    plt.title(f"AUC vs topology degree @ round {fixed_round} (all alphas)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    out_path = os.path.join(out_dir, "auc_vs_degree_all_alphas.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-node-per-round CSVs into AUC vs topology degree plots, "
            "averaging over nodes (within run) and over seeds (across runs).\n\n"
            "Expected CSV columns: round,node_id,train_acc,test_acc,auc\n"
            "Expected filename tokens: alphaX and regular_degree:Y (JSON-ish ok)."
        )
    )
    parser.add_argument(
        "--results-glob",
        type=str,
        default="../../results/*alpha*.csv",
        help="Glob pattern to find CSV result files.",
    )
    parser.add_argument(
        "--fixed-round",
        type=int,
        default=30,
        help="Round index to use; if not present, uses last usable round per file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../plots/alpha_by_degree",
        help="Directory to save aggregated CSVs and plots.",
    )
    parser.add_argument(
        "--no-combined-plot",
        action="store_true",
        help="Disable the combined plot with all alphas on one figure.",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.results_glob))
    if not files:
        raise SystemExit(f"No files matched pattern: {args.results_glob}")

    print(f"Found {len(files)} result files.")
    for f in files[:10]:
        print("  -", os.path.basename(f))
    if len(files) > 10:
        print("  ...")

    runs = load_runs(files, fixed_round=args.fixed_round)
    runs_df = pd.DataFrame(runs)

    # Save run-level table (useful for debugging what got picked)
    os.makedirs(args.output_dir, exist_ok=True)
    runs_df.to_csv(os.path.join(args.output_dir, "runs_picked_metrics.csv"), index=False)

    # Aggregate over seeds for each (alpha, degree)
    agg = aggregate_over_seeds(runs_df)
    agg.to_csv(os.path.join(args.output_dir, "auc_acc_by_alpha_degree.csv"), index=False)

    # Plot AUC vs degree per alpha
    plot_auc_vs_degree_per_alpha(agg, out_dir=args.output_dir, fixed_round=args.fixed_round)

    # Optional combined plot
    if not args.no_combined_plot:
        plot_auc_vs_degree_all_alphas(agg, out_dir=args.output_dir, fixed_round=args.fixed_round)

    print(f"\nSaved:\n  - {os.path.join(args.output_dir, 'runs_picked_metrics.csv')}"
          f"\n  - {os.path.join(args.output_dir, 'auc_acc_by_alpha_degree.csv')}"
          f"\n  - auc_vs_degree_alpha*.png (one per alpha)"
          f"\n  - auc_vs_degree_all_alphas.png (unless disabled)"
          f"\n\nOutput dir: {args.output_dir}")


if __name__ == "__main__":
    main()
