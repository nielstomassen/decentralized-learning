#!/usr/bin/env python3
"""
Plot privacy–utility tradeoff for the 2×2 hybrid ablation (DP × chunk).

Reads MIA result CSVs (with enable_dp, enable_chunking in filename or columns),
groups by (enable_dp, enable_chunking), and plots:
  - Mean test accuracy (utility) and mean MIA AUC (privacy leakage) per condition.
  - Optional: scatter accuracy vs AUC to visualize tradeoff (high acc + low AUC = better).

Usage:
  python hybrid_privacy_tradeoff.py --results-dir results/hybrid_ablation --out-dir plots/hybrid
  python hybrid_privacy_tradeoff.py --results-glob "results/hybrid_ablation/*.csv" --out-dir plots/hybrid
"""

import argparse
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parse dp/chunk/noise from filename (e.g. ..._dp1_chunk0_noise0.5_...)
DP_REGEX = re.compile(r"_dp([01])(?:_|$)", re.IGNORECASE)
CHUNK_REGEX = re.compile(r"_chunk([01])(?:_|$)", re.IGNORECASE)
NOISE_REGEX = re.compile(r"_noise([0-9.]+)(?:_|$)", re.IGNORECASE)


def parse_dp_chunk_from_path(path: str) -> dict:
    """Extract enable_dp, enable_chunking, dp_noise from filename."""
    base = os.path.basename(path)
    out = {"enable_dp": 0, "enable_chunking": 0, "dp_noise": 0.0}
    m = DP_REGEX.search(base)
    if m:
        out["enable_dp"] = int(m.group(1))
    m = CHUNK_REGEX.search(base)
    if m:
        out["enable_chunking"] = int(m.group(1))
    m = NOISE_REGEX.search(base)
    if m:
        out["dp_noise"] = float(m.group(1))
    return out


def load_and_label(path: str) -> pd.DataFrame:
    """Load CSV and ensure enable_dp, enable_chunking, dp_noise are set (from filename if missing)."""
    df = pd.read_csv(path)
    for col in ["round", "node_id", "global_test_acc", "max_auc", "avg_auc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    parsed = parse_dp_chunk_from_path(path)
    if "enable_dp" not in df.columns:
        df["enable_dp"] = parsed["enable_dp"]
    if "enable_chunking" not in df.columns:
        df["enable_chunking"] = parsed["enable_chunking"]
    if "dp_noise" not in df.columns:
        df["dp_noise"] = parsed["dp_noise"]
    df["_source_file"] = path
    return df


def condition_label(row) -> str:
    dp, ch = int(row["enable_dp"]), int(row["enable_chunking"])
    if dp and ch:
        return "DP + chunk (hybrid)"
    if dp:
        return "DP only"
    if ch:
        return "Chunk only"
    return "No DP, no chunk (baseline)"


def main():
    p = argparse.ArgumentParser(
        description="Plot privacy–utility tradeoff for hybrid (DP × chunk) ablation."
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing MIA CSV files (e.g. results/hybrid_ablation).",
    )
    p.add_argument(
        "--results-glob",
        type=str,
        default=None,
        help="Glob pattern for CSV files (e.g. 'results/hybrid_ablation/*.csv').",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="plots/hybrid",
        help="Output directory for plots.",
    )
    p.add_argument(
        "--round",
        type=int,
        default=None,
        help="Round to use for metrics (default: last round in each file).",
    )
    p.add_argument(
        "--auc-col",
        type=str,
        choices=["max_auc", "avg_auc"],
        default="max_auc",
        help="Which AUC column to use for privacy leakage.",
    )
    p.add_argument("--xmin", type=float, default=None, help="X axis lower limit (default: 0).")
    p.add_argument("--xmax", type=float, default=None, help="X axis upper limit (default: 1.05).")
    p.add_argument("--ymin", type=float, default=None, help="Y axis lower limit (default: 0).")
    p.add_argument("--ymax", type=float, default=None, help="Y axis upper limit (default: 1.05).")
    p.add_argument(
        "--xscale",
        type=str,
        choices=["linear", "log"],
        default="linear",
        help="X axis scale.",
    )
    p.add_argument(
        "--yscale",
        type=str,
        choices=["linear", "log"],
        default="linear",
        help="Y axis scale.",
    )
    args = p.parse_args()

    xmin = 0.0 if args.xmin is None else args.xmin
    xmax = 1.05 if args.xmax is None else args.xmax
    ymin = 0.0 if args.ymin is None else args.ymin
    ymax = 1.05 if args.ymax is None else args.ymax

    if args.results_glob:
        import glob
        paths = sorted(glob.glob(args.results_glob))
    elif args.results_dir:
        paths = sorted(
            os.path.join(args.results_dir, f)
            for f in os.listdir(args.results_dir)
            if f.endswith(".csv")
        )
    else:
        raise ValueError("Provide either --results-dir or --results-glob")

    if not paths:
        raise FileNotFoundError("No CSV files found")

    frames = []
    for path in paths:
        try:
            df = load_and_label(path)
            frames.append(df)
        except Exception as e:
            print(f"Warning: skip {path}: {e}")
            continue

    if not frames:
        raise ValueError("No CSVs loaded")

    full = pd.concat(frames, ignore_index=True)
    full["condition"] = full.apply(condition_label, axis=1)

    # Per-file, per-condition: take target round or last round, then mean over nodes
    target_round = args.round
    rows = []
    for (src, enable_dp, enable_chunking), g in full.groupby(
        ["_source_file", "enable_dp", "enable_chunking"]
    ):
        if target_round is not None:
            rdf = g[g["round"] == target_round]
        else:
            last_r = g["round"].max()
            rdf = g[g["round"] == last_r]
        if rdf.empty:
            continue
        acc = rdf["global_test_acc"].mean()
        auc = rdf[args.auc_col].mean()
        rows.append(
            {
                "enable_dp": enable_dp,
                "enable_chunking": enable_chunking,
                "condition": condition_label(rdf.iloc[0]),
                "mean_accuracy": acc,
                "mean_auc": auc,
                "n_nodes": len(rdf),
            }
        )

    by_cond = pd.DataFrame(rows).groupby(
        ["enable_dp", "enable_chunking", "condition"], as_index=False
    ).agg(
        mean_accuracy=("mean_accuracy", "mean"),
        std_accuracy=("mean_accuracy", "std"),
        mean_auc=("mean_auc", "mean"),
        std_auc=("mean_auc", "std"),
        n_seeds=("mean_accuracy", "count"),
    )

    # Stable order: baseline, DP only, Chunk only, hybrid
    order = [
        "No DP, no chunk (baseline)",
        "DP only",
        "Chunk only",
        "DP + chunk (hybrid)",
    ]
    by_cond["condition"] = pd.Categorical(by_cond["condition"], categories=order, ordered=True)
    by_cond = by_cond.sort_values("condition")

    os.makedirs(args.out_dir, exist_ok=True)

    # Figure 1: Bar chart – accuracy and AUC by condition
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(by_cond))
    w = 0.35

    acc_vals = by_cond["mean_accuracy"].values
    acc_err = by_cond["std_accuracy"].fillna(0).values
    ax1.bar(x - w / 2, acc_vals, w, yerr=acc_err, capsize=3, label="Test accuracy", color="tab:blue")
    ax1.set_ylabel("Mean test accuracy (utility)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(by_cond["condition"], rotation=15, ha="right")
    ax1.set_ylim(ymin, ymax)
    ax1.set_yscale(args.yscale)
    ax1.legend()

    auc_vals = by_cond["mean_auc"].values
    auc_err = by_cond["std_auc"].fillna(0).values
    ax2.bar(x + w / 2, auc_vals, w, yerr=auc_err, capsize=3, label=f"MIA {args.auc_col}", color="tab:red")
    ax2.set_ylabel(f"Mean MIA {args.auc_col} (leakage)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(by_cond["condition"], rotation=15, ha="right")
    ax2.set_ylim(0.5, 0.8)
    ax2.set_yscale(args.yscale)
    ax2.legend()

    n_seeds = int(by_cond["n_seeds"].max()) if len(by_cond) > 0 else 0
    seed_note = f" (mean ± std over {n_seeds} seeds)" if n_seeds > 0 else ""
    fig.suptitle(f"Hybrid privacy ablation: utility vs privacy leakage{seed_note}")
    plt.tight_layout()
    bar_path = os.path.join(args.out_dir, "hybrid_ablation_bars.png")
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {bar_path}")

    # Figure 2: Scatter – accuracy vs AUC (one point per condition); lower AUC = more private
    fig, ax = plt.subplots(figsize=(6, 5))
    for _, row in by_cond.iterrows():
        ax.scatter(
            row["mean_accuracy"],
            row["mean_auc"],
            s=120,
            label=row["condition"],
        )
        ax.annotate(
            row["condition"],
            (row["mean_accuracy"], row["mean_auc"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    n_seeds = int(by_cond["n_seeds"].max()) if len(by_cond) > 0 else 0
    ax.set_xlabel("Mean test accuracy (utility)")
    ax.set_ylabel(f"Mean MIA {args.auc_col} (leakage)")
    ax.set_title(f"Privacy–utility tradeoff (mean over {n_seeds} seeds; ideal: high accuracy, low AUC)")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.5, 0.8)
    ax.set_xscale(args.xscale)
    ax.set_yscale(args.yscale)
    ax.grid(True, alpha=0.3)
    scatter_path = os.path.join(args.out_dir, "hybrid_privacy_utility_scatter.png")
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {scatter_path}")


if __name__ == "__main__":
    main()
