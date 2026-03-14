#!/usr/bin/env python3
"""
Analyze hybrid (DP + chunk) sweep over noise multiplier and max_grad_norm.

Reads MIA result CSVs from the noise/clip sweep, keeps only hybrid runs (dp1, chunk1),
groups by (dp_noise, dp_max_grad_norm), and produces:
  - summary CSV: mean ± std test accuracy and mean ± std MIA AUC per (noise, clip)
  - scatter: accuracy vs AUC with points labeled by (noise, clip) to find sweet spot
  - optional bar chart comparing accuracy and AUC across (noise, clip) configs

Usage:
  python scripts/analyze_hybrid_noise_clip_sweep.py --results-dir results/cifar100/hybrid_noise_clip_sweep/er_p_0.08 --out-dir plots/hybrid_noise_clip_sweep
  python scripts/analyze_hybrid_noise_clip_sweep.py --results-glob "results/cifar100/hybrid_noise_clip_sweep/**/*.csv" --out-dir plots/hybrid_noise_clip_sweep
"""

import argparse
import os
import re
import glob as glob_module

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# From filename: ..._dp1_chunk1_..._noise0.5_clip1.0_seed4235.csv (clip optional for old runs)
DP_REGEX = re.compile(r"_dp([01])(?:_|$)", re.IGNORECASE)
CHUNK_REGEX = re.compile(r"_chunk([01])(?:_|$)", re.IGNORECASE)
NOISE_REGEX = re.compile(r"_noise([0-9.]+)(?:_|$)", re.IGNORECASE)
CLIP_REGEX = re.compile(r"_clip([0-9.]+)(?:_|$)", re.IGNORECASE)


def parse_from_path(path: str) -> dict:
    """Extract enable_dp, enable_chunking, dp_noise, dp_max_grad_norm from filename."""
    base = os.path.basename(path)
    out = {"enable_dp": 0, "enable_chunking": 0, "dp_noise": 0.0, "dp_max_grad_norm": 1.0}
    m = DP_REGEX.search(base)
    if m:
        out["enable_dp"] = int(m.group(1))
    m = CHUNK_REGEX.search(base)
    if m:
        out["enable_chunking"] = int(m.group(1))
    m = NOISE_REGEX.search(base)
    if m:
        out["dp_noise"] = float(m.group(1))
    m = CLIP_REGEX.search(base)
    if m:
        out["dp_max_grad_norm"] = float(m.group(1))
    return out


def load_and_label(path: str) -> pd.DataFrame:
    """Load CSV; ensure dp/chunk/noise/clip from filename if missing in columns."""
    df = pd.read_csv(path)
    for col in ["round", "node_id", "global_test_acc", "max_auc", "avg_auc", "dp_noise", "dp_max_grad_norm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    parsed = parse_from_path(path)
    if "enable_dp" not in df.columns:
        df["enable_dp"] = parsed["enable_dp"]
    if "enable_chunking" not in df.columns:
        df["enable_chunking"] = parsed["enable_chunking"]
    if "dp_noise" not in df.columns or df["dp_noise"].isna().all():
        df["dp_noise"] = parsed["dp_noise"]
    if "dp_max_grad_norm" not in df.columns or df["dp_max_grad_norm"].isna().all():
        df["dp_max_grad_norm"] = parsed["dp_max_grad_norm"]
    df["_source_file"] = path
    return df


def main():
    p = argparse.ArgumentParser(
        description="Analyze hybrid noise × max_grad_norm sweep (utility vs privacy)."
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing MIA CSV files from the sweep.",
    )
    p.add_argument(
        "--results-glob",
        type=str,
        default=None,
        help="Glob pattern for CSV files (e.g. 'results/**/hybrid_noise_clip_sweep/**/*.csv').",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="plots/hybrid_noise_clip_sweep",
        help="Output directory for summary CSV and plots.",
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
    p.add_argument(
        "--no-bars",
        action="store_true",
        help="Skip bar chart (accuracy and AUC by config).",
    )
    args = p.parse_args()

    if args.results_glob:
        paths = sorted(glob_module.glob(args.results_glob))
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
            # Keep only hybrid (DP + chunk)
            if (df["enable_dp"].iloc[0] != 1) or (df["enable_chunking"].iloc[0] != 1):
                continue
            frames.append(df)
        except Exception as e:
            print(f"Warning: skip {path}: {e}")
            continue

    if not frames:
        raise ValueError("No hybrid (dp1, chunk1) CSVs loaded")

    full = pd.concat(frames, ignore_index=True)
    target_round = args.round

    # Per-file: last round (or target round), then mean over nodes
    rows = []
    for (src, dp_noise, dp_clip), g in full.groupby(
        ["_source_file", "dp_noise", "dp_max_grad_norm"]
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
                "dp_noise": dp_noise,
                "dp_max_grad_norm": dp_clip,
                "mean_accuracy": acc,
                "mean_auc": auc,
                "n_nodes": len(rdf),
            }
        )

    run_df = pd.DataFrame(rows)
    # Aggregate over seeds (same noise/clip can appear from different files = seeds)
    by_config = run_df.groupby(
        ["dp_noise", "dp_max_grad_norm"], as_index=False
    ).agg(
        mean_accuracy=("mean_accuracy", "mean"),
        std_accuracy=("mean_accuracy", "std"),
        mean_auc=("mean_auc", "mean"),
        std_auc=("mean_auc", "std"),
        n_runs=("mean_accuracy", "count"),
    )
    by_config = by_config.sort_values(["dp_noise", "dp_max_grad_norm"])

    os.makedirs(args.out_dir, exist_ok=True)

    # Summary CSV
    summary_path = os.path.join(args.out_dir, "hybrid_noise_clip_sweep_summary.csv")
    by_config.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
    print(by_config.to_string(index=False))

    # Scatter: accuracy vs AUC, one point per (noise, clip); ideal = high acc, low AUC
    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in by_config.iterrows():
        label = f"σ={row['dp_noise']}, C={row['dp_max_grad_norm']}"
        ax.scatter(
            row["mean_accuracy"],
            row["mean_auc"],
            s=100,
            label=label,
        )
        ax.annotate(
            label,
            (row["mean_accuracy"], row["mean_auc"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Mean test accuracy (utility)")
    ax.set_ylabel(f"Mean MIA {args.auc_col} (leakage)")
    ax.set_title("Hybrid noise × clip sweep (ideal: high accuracy, low AUC)")
    ax.legend(loc="best", fontsize=7)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.1, 1.0)
    ax.grid(True, alpha=0.3)
    scatter_path = os.path.join(args.out_dir, "hybrid_noise_clip_sweep_scatter.png")
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter: {scatter_path}")

    # Bar chart: accuracy and AUC by config
    if not args.no_bars:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        x = np.arange(len(by_config))
        w = 0.35
        labels = [f"σ={r['dp_noise']}\nC={r['dp_max_grad_norm']}" for _, r in by_config.iterrows()]
        acc_vals = by_config["mean_accuracy"].values
        acc_err = by_config["std_accuracy"].fillna(0).values
        ax1.bar(x - w / 2, acc_vals, w, yerr=acc_err, capsize=3, color="tab:blue")
        ax1.set_ylabel("Mean test accuracy (utility)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=15, ha="right")
        ax1.set_ylim(0, 1.05)
        auc_vals = by_config["mean_auc"].values
        auc_err = by_config["std_auc"].fillna(0).values
        ax2.bar(x + w / 2, auc_vals, w, yerr=auc_err, capsize=3, color="tab:red")
        ax2.set_ylabel(f"Mean MIA {args.auc_col} (leakage)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=15, ha="right")
        ax2.set_ylim(0.4, 1.0)
        fig2.suptitle("Hybrid sweep: utility and privacy by (noise σ, clip C)")
        plt.tight_layout()
        bar_path = os.path.join(args.out_dir, "hybrid_noise_clip_sweep_bars.png")
        fig2.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved bars: {bar_path}")

    print("Done.")


if __name__ == "__main__":
    main()
