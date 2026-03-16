#!/usr/bin/env python3
"""
Plot privacy–utility tradeoff for the 2×2 hybrid ablation (DP × chunk).

Reads MIA result CSVs (with enable_dp, enable_chunking in filename or columns),
groups by (enable_dp, enable_chunking), and plots:
  - Bar chart: mean ± std test accuracy and MIA AUC per condition.
  - Scatter: accuracy vs AUC (ideal: high acc, low AUC).
  - Box plots: node-level accuracy and AUC by condition (fixed order), with mean ± std overlay.
  - Score plot: bar chart of score = u - λ*r per condition, ordered by score (best first).
  - Optional --order-by-score: order conditions by score = u - λ*r (r = max(0, 2*AUC-1)); --lambda sets λ.

Usage:
  python hybrid_privacy_tradeoff.py --results-dir results/hybrid_ablation --out-dir plots/hybrid
  python hybrid_privacy_tradeoff.py --results-glob "results/hybrid_ablation/*.csv" --out-dir plots/hybrid --order-by-score --lambda 1
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
    p.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=1.0,
        help="Weight for privacy risk in score: score = u - λ*r, r = max(0, 2*AUC-1). Used only with --order-by-score.",
    )
    p.add_argument(
        "--order-by-score",
        action="store_true",
        help="Order conditions by score (u - λ*r) descending; best first.",
    )
    p.add_argument(
        "--no-boxplots",
        action="store_true",
        help="Skip box plots (node-level accuracy and AUC by condition with mean ± std).",
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

    # Fixed order for box plot; score order used for bars/scatter when --order-by-score
    fixed_order = [
        "No DP, no chunk (baseline)",
        "DP only",
        "Chunk only",
        "DP + chunk (hybrid)",
    ]
    order = list(fixed_order)
    # Optional: score = u - λ*r, r = max(0, 2*AUC-1); order by score descending
    lam = getattr(args, "lambda_", 1.0)
    by_cond["privacy_risk"] = np.maximum(0, 2 * by_cond["mean_auc"] - 1)
    by_cond["score"] = by_cond["mean_accuracy"] - lam * by_cond["privacy_risk"]
    if getattr(args, "order_by_score", False):
        by_cond = by_cond.sort_values("score", ascending=False).reset_index(drop=True)
        order = by_cond["condition"].tolist()
    else:
        by_cond["condition"] = pd.Categorical(by_cond["condition"], categories=order, ordered=True)
        by_cond = by_cond.sort_values("condition").reset_index(drop=True)
    by_cond["condition"] = pd.Categorical(by_cond["condition"], categories=order, ordered=True)

    # Node-level data for target round (for box plots)
    if target_round is not None:
        round_df = full[full["round"] == target_round].copy()
    else:
        last_round = full.groupby("_source_file")["round"].transform("max")
        round_df = full[full["round"] == last_round].copy()
    round_df["_acc"] = round_df["global_test_acc"]
    round_df["_auc"] = round_df[args.auc_col]

    os.makedirs(args.out_dir, exist_ok=True)

    # Summary CSV with score (so you can see score per condition)
    summary_path = os.path.join(args.out_dir, "hybrid_ablation_summary.csv")
    by_cond.to_csv(summary_path, index=False)
    print(f"Saved summary (with score): {summary_path}")

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
    ax2.set_ylim(0.4, 1)
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
    ax.set_ylim(0.1, 1)
    ax.set_xscale(args.xscale)
    ax.set_yscale(args.yscale)
    ax.grid(True, alpha=0.3)
    scatter_path = os.path.join(args.out_dir, "hybrid_privacy_utility_scatter.png")
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {scatter_path}")

    # Figure 3: Box plots – fixed order (baseline, DP only, Chunk only, hybrid); boxes + mean ± std overlay
    if not getattr(args, "no_boxplots", False):
        cond_order = list(fixed_order)
        acc_by_cond = [round_df.loc[round_df["condition"] == c, "_acc"].dropna().values for c in cond_order]
        auc_by_cond = [round_df.loc[round_df["condition"] == c, "_auc"].dropna().values for c in cond_order]
        pos = np.arange(len(cond_order))
        fig3, (bx1, bx2) = plt.subplots(1, 2, figsize=(10, 4))
        bp1 = bx1.boxplot(acc_by_cond, positions=pos, widths=0.5, patch_artist=True, showfliers=False)
        bp2 = bx2.boxplot(auc_by_cond, positions=pos, widths=0.5, patch_artist=True, showfliers=False)
        for patch in bp1["boxes"]:
            patch.set_facecolor("tab:blue")
            patch.set_alpha(0.6)
        for patch in bp2["boxes"]:
            patch.set_facecolor("tab:red")
            patch.set_alpha(0.6)
        for i, c in enumerate(cond_order):
            row = by_cond[by_cond["condition"] == c].iloc[0]
            mu_a = row["mean_accuracy"]
            s_a = row["std_accuracy"] if pd.notna(row["std_accuracy"]) else 0
            mu_u = row["mean_auc"]
            s_u = row["std_auc"] if pd.notna(row["std_auc"]) else 0
            bx1.plot([i], [mu_a], "k*", markersize=10)
            bx1.errorbar([i], [mu_a], yerr=[s_a], fmt="none", color="black", capsize=3)
            bx2.plot([i], [mu_u], "k*", markersize=10)
            bx2.errorbar([i], [mu_u], yerr=[s_u], fmt="none", color="black", capsize=3)
            # One outlier per box: the max (worst) value
            if len(acc_by_cond[i]) > 0:
                bx1.plot(i, np.max(acc_by_cond[i]), "o", color="tab:blue", markersize=5, alpha=0.8, zorder=5)
            if len(auc_by_cond[i]) > 0:
                bx2.plot(i, np.max(auc_by_cond[i]), "o", color="tab:red", markersize=5, alpha=0.8, zorder=5)
        bx1.set_xticks(pos)
        bx1.set_xticklabels(cond_order, rotation=15, ha="right")
        bx1.set_ylabel("Test accuracy (node-level)")
        bx1.set_ylim(ymin, ymax)
        bx2.set_xticks(pos)
        bx2.set_xticklabels(cond_order, rotation=15, ha="right")
        bx2.set_ylabel(f"MIA {args.auc_col} (node-level)")
        bx2.set_ylim(0.4, 1)
        fig3.suptitle("Hybrid ablation: box plots (node-level); star = mean, error bar = std, circle = max (one outlier per box)")
        plt.tight_layout()
        box_path = os.path.join(args.out_dir, "hybrid_ablation_boxplots.png")
        fig3.savefig(box_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {box_path}")

    # Figure 4: Score bar chart – conditions ordered by score (best first), like the other script
    by_cond_score = by_cond.sort_values("score", ascending=False).reset_index(drop=True)
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(by_cond_score))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(by_cond_score))[::-1])
    ax4.bar(x, by_cond_score["score"].values, color=colors)
    ax4.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax4.set_ylabel(f"Score (u − λr, r = max(0, 2·AUC−1)); λ = {lam}; higher = better")
    ax4.set_xticks(x)
    ax4.set_xticklabels(by_cond_score["condition"].values, rotation=15, ha="right")
    ax4.set_title("Condition order by score (best first)")
    plt.tight_layout()
    score_path = os.path.join(args.out_dir, "hybrid_ablation_score.png")
    fig4.savefig(score_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {score_path}")


if __name__ == "__main__":
    main()
