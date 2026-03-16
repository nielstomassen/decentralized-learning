#!/usr/bin/env python3
"""
Plot Pareto frontier for hybrid noise × clip sweep (utility vs privacy).

Reads the summary CSV produced by analyze_hybrid_noise_clip_sweep.py (columns:
mean_accuracy, mean_auc, dp_noise, dp_max_grad_norm). We maximize accuracy (utility)
and minimize AUC (leakage). A point is Pareto-optimal if no other point has both
higher-or-equal accuracy and lower-or-equal AUC with at least one strict improvement.

Outputs a scatter of all configs with the Pareto frontier highlighted and connected.

Usage:
  python scripts/plot_hybrid_noise_sweep_pareto.py --summary-csv plots/hybrid_noise_clip_sweep/hybrid_noise_clip_sweep_summary.csv --out-dir plots/hybrid_noise_clip_sweep
  python scripts/plot_hybrid_noise_sweep_pareto.py -s synced/plots/hybrid_noise_clip_sweep/hybrid_noise_clip_sweep_summary.csv -o synced/plots/hybrid_noise_clip_sweep
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects


def compute_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pareto frontier for (maximize accuracy, minimize AUC).

    Points are sorted by accuracy descending; a point is on the frontier if
    when walking from high to low accuracy, its AUC is strictly lower than
    the previous frontier point (lower envelope).
    """
    df = df.sort_values(["mean_accuracy", "mean_auc"], ascending=[False, True]).reset_index(
        drop=True
    )
    frontier_indices = []
    best_auc_so_far = np.inf
    for i, row in df.iterrows():
        if row["mean_auc"] < best_auc_so_far:
            frontier_indices.append(i)
            best_auc_so_far = row["mean_auc"]
    return df.loc[frontier_indices].sort_values("mean_accuracy", ascending=True).reset_index(
        drop=True
    )


def main():
    p = argparse.ArgumentParser(
        description="Plot Pareto frontier for hybrid noise × clip sweep (utility vs leakage)."
    )
    p.add_argument(
        "-s",
        "--summary-csv",
        type=str,
        required=True,
        help="Path to hybrid_noise_clip_sweep_summary.csv (from analyze_hybrid_noise_clip_sweep.py).",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for plot (default: same dir as summary CSV).",
    )
    p.add_argument(
        "--auc-col",
        type=str,
        choices=["mean_auc", "max_auc", "avg_auc"],
        default="mean_auc",
        help="Column name for leakage metric (summary CSV has mean_auc). Default: mean_auc.",
    )
    p.add_argument(
        "--acc-col",
        type=str,
        default="mean_accuracy",
        help="Column name for utility. Default: mean_accuracy.",
    )
    p.add_argument(
        "--no-frontier-line",
        action="store_true",
        help="Do not draw the step line connecting Pareto points.",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title (default: auto).",
    )
    p.add_argument(
        "--xmin", type=float, default=None, help="X axis min (default: data range with margin)."
    )
    p.add_argument(
        "--xmax", type=float, default=None, help="X axis max (default: data range with margin)."
    )
    p.add_argument(
        "--ymin", type=float, default=None, help="Y axis min (default: data range with margin)."
    )
    p.add_argument(
        "--ymax", type=float, default=None, help="Y axis max (default: data range with margin)."
    )
    p.add_argument(
        "--margin",
        type=float,
        default=0.02,
        help="Fraction of data range to add as padding when auto-scaling (default: 0.02).",
    )
    args = p.parse_args()

    if not os.path.isfile(args.summary_csv):
        raise FileNotFoundError(f"Summary CSV not found: {args.summary_csv}")

    df = pd.read_csv(args.summary_csv)
    for col in [args.acc_col, args.auc_col, "dp_noise", "dp_max_grad_norm"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in summary CSV; got {list(df.columns)}")

    # Use the column names from CSV (summary has mean_accuracy, mean_auc)
    acc_col = args.acc_col
    auc_col = args.auc_col

    frontier = compute_pareto_frontier(df)

    # Assign each config a number (order: accuracy desc, then AUC asc) for the key
    df_sorted = df.sort_values([acc_col, auc_col], ascending=[False, True]).reset_index(drop=True)
    df_sorted["_num"] = np.arange(1, len(df_sorted) + 1)
    num_map = df_sorted.set_index(["dp_noise", "dp_max_grad_norm"])["_num"].to_dict()
    df["_num"] = df.apply(lambda r: num_map.get((r["dp_noise"], r["dp_max_grad_norm"]), 0), axis=1)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.summary_csv))
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    # All points (dominated)
    ax.scatter(
        df[acc_col],
        df[auc_col],
        s=80,
        c="lightgray",
        alpha=0.8,
        edgecolors="gray",
        linewidths=0.5,
        label="Other configs",
        zorder=1,
    )

    # Pareto frontier line (step: horizontal then vertical between consecutive points)
    if not args.no_frontier_line and len(frontier) >= 2:
        acc = frontier[acc_col].values
        auc = frontier[auc_col].values
        # Step: (a0,u0) -> (a1,u0) -> (a1,u1) -> (a2,u1) -> ... -> (a_{n-1}, u_n)
        step_x = np.concatenate([[acc[0]], np.repeat(acc[1:], 2)])
        step_y = np.concatenate([np.repeat(auc[:-1], 2), [auc[-1]]])
        ax.plot(step_x, step_y, color="C0", linewidth=2, alpha=0.9, zorder=2)

    # Pareto-optimal points
    ax.scatter(
        frontier[acc_col],
        frontier[auc_col],
        s=120,
        c="C0",
        alpha=0.9,
        edgecolors="darkblue",
        linewidths=1.5,
        label="Pareto frontier",
        zorder=3,
    )

    # Number each point so you can match to the key (white outline for readability)
    for _, row in df.iterrows():
        ax.annotate(
            str(int(row["_num"])),
            (row[acc_col], row[auc_col]),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            path_effects=[
                patheffects.withStroke(linewidth=2.5, foreground="white")
            ],
            zorder=4,
        )

    # Key: number -> (σ, C) for every config so you can identify points
    key_lines = []
    for _, row in df_sorted.iterrows():
        key_lines.append(f"{int(row['_num'])}: σ={row['dp_noise']}, C={row['dp_max_grad_norm']}")
    textstr = "Config key:\n" + "\n".join(key_lines)
    props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.9)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
        family="monospace",
    )

    ax.set_xlabel("Mean test accuracy (utility)", fontsize=11)
    ax.set_ylabel("Mean MIA AUC (leakage)", fontsize=11)
    ax.set_title(
        args.title
        or "Hybrid noise × clip sweep — Pareto frontier (ideal: high accuracy, low AUC)",
        fontsize=12,
    )

    # Scale axes to data range so points are spread out (unless user set limits)
    acc_vals = df[acc_col].values
    auc_vals = df[auc_col].values
    margin = args.margin
    x_range = acc_vals.max() - acc_vals.min() or 0.01
    y_range = auc_vals.max() - auc_vals.min() or 0.01
    xmin = args.xmin if args.xmin is not None else max(0, acc_vals.min() - margin * x_range)
    xmax = args.xmax if args.xmax is not None else min(1, acc_vals.max() + margin * x_range)
    ymin = args.ymin if args.ymin is not None else max(0.1, auc_vals.min() - margin * y_range)
    ymax = args.ymax if args.ymax is not None else min(1, auc_vals.max() + margin * y_range)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "hybrid_noise_clip_sweep_pareto.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Pareto plot: {out_path}")
    print("Pareto-optimal configs:")
    print(frontier[["dp_noise", "dp_max_grad_norm", acc_col, auc_col]].to_string(index=False))


if __name__ == "__main__":
    main()
