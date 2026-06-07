#!/usr/bin/env python3
"""
Plot peer-count sweep used in the Appendix experiment.

Loads CSVs from ``<results-root>/er_p_<p>/peers_<n>/`` and produces hybrid-ablation-style
figures with peer count as an additional grouping dimension:
  - Grouped bars: accuracy / MIA AUC by condition, bars grouped by peer count.
  - Line plots: metric vs peer count (one line per condition) — highlights peer-count effect.
  - Scatter: accuracy vs AUC (marker = peer count, color = condition).
  - Deployment scores S_λ for λ∈{0.25, 0.5, 0.75}, faceted by condition.

Usage:
  python3 -m plotting.appendix.peer_count_mechanism_sweep \
      --results-root results/appendix/peer_count_mechanism_sweep/er_p_0.08 \
      --out-dir plots/appendix/peer_count_mechanism_sweep/er_p_0.08
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plotting.hybrid_ablation.chunkdp_labels import (
    CHUNKDP_CONDITION,
    auc_metric_title_suffix,
    chunkdp_xtick_label,
    mean_accuracy_axis_label,
    mean_mia_auc_axis_label,
    normalize_condition_label,
)
from plotting.hybrid_ablation.hybrid_lambda_deployment_scores import (
    DEFAULT_LAMBDAS,
    deployment_score,
    export_ablation_lambda_table,
)
from plotting.hybrid_ablation.hybrid_privacy_tradeoff import (
    _save_fig,
    condition_label,
    load_and_label,
)

CONDITION_ORDER = [
    "No DP, no chunk (baseline)",
    "DP only",
    "Topology-aware chunking",
    CHUNKDP_CONDITION,
]

# Appendix line plot: baseline vs topology-aware chunking only.
LINE_PLOT_CONDITIONS = [
    "No DP, no chunk (baseline)",
    "Topology-aware chunking",
]

LINE_PLOT_STYLE = {
    "No DP, no chunk (baseline)": {"color": "#2c5282", "marker": "o"},
    "Topology-aware chunking": {"color": "#c05621", "marker": "s"},
}

PEER_COLORS = {
    10: "#1f77b4",
    25: "#ff7f0e",
    50: "#2ca02c",
    75: "#d62728",
}

PEER_MARKERS = {
    10: "o",
    25: "s",
    50: "^",
    75: "D",
}


def _parse_peers_from_path(path: str) -> int | None:
    m = re.search(r"/peers_(\d+)/", path.replace("\\", "/"))
    if m:
        return int(m.group(1))
    m = re.search(r"peers_(\d+)", os.path.dirname(path))
    return int(m.group(1)) if m else None


def discover_peer_dirs(results_root: str) -> list[tuple[int, str]]:
    root = Path(results_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Results root not found: {results_root}")
    out: list[tuple[int, str]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        m = re.fullmatch(r"peers_(\d+)", child.name)
        if not m:
            continue
        peers = int(m.group(1))
        if any(child.glob("*.csv")):
            out.append((peers, str(child)))
    if not out:
        raise FileNotFoundError(f"No peers_* CSV directories under {results_root}")
    return sorted(out, key=lambda x: x[0])


def aggregate_sweep(
    peer_dirs: list[tuple[int, str]],
    auc_col: str,
    target_round: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary by peers×condition, node-level long df for last/target round)."""
    frames: list[pd.DataFrame] = []
    for peers, pdir in peer_dirs:
        for fname in sorted(os.listdir(pdir)):
            if not fname.endswith(".csv"):
                continue
            path = os.path.join(pdir, fname)
            try:
                df = load_and_label(path)
            except Exception as exc:
                print(f"Warning: skip {path}: {exc}", file=sys.stderr)
                continue
            df["peers"] = peers
            df["condition"] = df.apply(condition_label, axis=1)
            frames.append(df)

    if not frames:
        raise ValueError("No CSVs loaded")

    full = pd.concat(frames, ignore_index=True)
    full["condition"] = full["condition"].map(normalize_condition_label)

    rows: list[dict] = []
    for (peers, src, enable_dp, enable_chunking), g in full.groupby(
        ["peers", "_source_file", "enable_dp", "enable_chunking"]
    ):
        if target_round is not None:
            rdf = g[g["round"] == target_round]
        else:
            last_r = g["round"].max()
            rdf = g[g["round"] == last_r]
        if rdf.empty:
            continue
        rows.append(
            {
                "peers": peers,
                "enable_dp": enable_dp,
                "enable_chunking": enable_chunking,
                "condition": normalize_condition_label(condition_label(rdf.iloc[0])),
                "mean_accuracy": rdf["global_test_acc"].mean(),
                "mean_auc": rdf[auc_col].mean(),
                "std_acc_nodes": rdf["global_test_acc"].std(),
                "std_auc_nodes": rdf[auc_col].std(),
            }
        )

    per_run = pd.DataFrame(rows)
    summary = (
        per_run.groupby(["peers", "enable_dp", "enable_chunking", "condition"], as_index=False)
        .agg(
            mean_accuracy=("mean_accuracy", "mean"),
            std_accuracy_across_seeds=("mean_accuracy", "std"),
            mean_node_std_accuracy=("std_acc_nodes", "mean"),
            mean_auc=("mean_auc", "mean"),
            std_auc_across_seeds=("mean_auc", "std"),
            mean_node_std_auc=("std_auc_nodes", "mean"),
            n_seeds=("mean_accuracy", "count"),
        )
    )

    if target_round is not None:
        round_df = full[full["round"] == target_round].copy()
    else:
        last_round = full.groupby("_source_file")["round"].transform("max")
        round_df = full[full["round"] == last_round].copy()
    round_df["_acc"] = round_df["global_test_acc"]
    round_df["_auc"] = round_df[auc_col]

    return summary, round_df


def _condition_categorical(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in CONDITION_ORDER if c in df["condition"].values]
    extra = sorted(set(df["condition"].astype(str)) - set(present))
    order = present + extra
    out = df.copy()
    out["condition"] = pd.Categorical(out["condition"], categories=order, ordered=True)
    return out.sort_values(["condition", "peers"]).reset_index(drop=True)


def plot_grouped_bars(
    summary: pd.DataFrame,
    out_path: str,
    auc_col: str,
    ymin: float,
    ymax: float,
) -> None:
    """Hybrid-ablation-style bars: x = condition, grouped bars = peer count."""
    summary = _condition_categorical(summary)
    conditions = summary["condition"].cat.categories.tolist()
    peer_counts = sorted(summary["peers"].unique())
    n_cond = len(conditions)
    n_peers = len(peer_counts)
    x = np.arange(n_cond)
    group_w = 0.78
    bar_w = group_w / n_peers

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for j, peers in enumerate(peer_counts):
        sub = summary[summary["peers"] == peers].set_index("condition").reindex(conditions)
        offset = (j - (n_peers - 1) / 2) * bar_w
        color = PEER_COLORS.get(peers, f"C{j}")
        ax1.bar(
            x + offset,
            sub["mean_accuracy"].values,
            bar_w,
            yerr=sub["mean_node_std_accuracy"].fillna(0).values,
            capsize=2,
            label=f"{peers} peers",
            color=color,
            edgecolor="black",
            linewidth=0.35,
        )
        ax2.bar(
            x + offset,
            sub["mean_auc"].values,
            bar_w,
            yerr=sub["mean_node_std_auc"].fillna(0).values,
            capsize=2,
            label=f"{peers} peers",
            color=color,
            edgecolor="black",
            linewidth=0.35,
        )

    display_labels = [chunkdp_xtick_label(c) for c in conditions]
    for ax, ylabel, ylim_top in (
        (ax1, mean_accuracy_axis_label(), ymax),
        (ax2, mean_mia_auc_axis_label(auc_col), 1.0),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(title="Peer count", fontsize=8, title_fontsize=8, loc="best")
    ax1.set_ylim(ymin, ylim_top)
    ax2.set_ylim(0.4, 1.0)

    plt.tight_layout()
    _save_fig(fig, out_path)
    plt.close()


def plot_peer_count_lines(
    summary: pd.DataFrame,
    out_path: str,
    auc_col: str,
    ymin: float,
    ymax: float,
) -> None:
    """Line plots: x = peer count, one line per condition (peer-count effect)."""
    summary = _condition_categorical(summary)
    summary = summary[summary["condition"].astype(str).isin(LINE_PLOT_CONDITIONS)].copy()
    peer_counts = sorted(summary["peers"].unique())
    conditions = [c for c in LINE_PLOT_CONDITIONS if c in summary["condition"].astype(str).values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for cond in conditions:
        sub = summary[summary["condition"] == cond].sort_values("peers")
        if sub.empty:
            continue
        label = chunkdp_xtick_label(cond)
        style = LINE_PLOT_STYLE.get(cond, {"color": "gray", "marker": "o"})
        plot_kwargs = dict(
            marker=style["marker"],
            capsize=3,
            label=label,
            color=style["color"],
            linewidth=2.0,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        ax1.errorbar(sub["peers"], sub["mean_accuracy"], yerr=sub["mean_node_std_accuracy"].fillna(0), **plot_kwargs)
        ax2.errorbar(sub["peers"], sub["mean_auc"], yerr=sub["mean_node_std_auc"].fillna(0), **plot_kwargs)

    for ax, ylabel, ylim_top in (
        (ax1, mean_accuracy_axis_label(), ymax),
        (ax2, mean_mia_auc_axis_label(auc_col), 1.0),
    ):
        ax.set_xlabel("Number of peers")
        ax.set_ylabel(ylabel)
        ax.set_xticks(peer_counts)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.5, loc="best", framealpha=0.95)
    ax1.set_ylim(ymin, ylim_top)
    ax2.set_ylim(0.4, 1.0)

    plt.tight_layout()
    _save_fig(fig, out_path)
    plt.close()


def plot_scatter(summary: pd.DataFrame, out_path: str, auc_col: str, xmin: float, xmax: float) -> None:
    summary = _condition_categorical(summary)
    conditions = summary["condition"].cat.categories.tolist()
    cmap = plt.cm.tab10(np.linspace(0, 0.9, len(conditions)))
    cond_color = {c: cmap[i] for i, c in enumerate(conditions)}

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for _, row in summary.iterrows():
        peers = int(row["peers"])
        cond = row["condition"]
        ax.scatter(
            row["mean_accuracy"],
            row["mean_auc"],
            s=90,
            color=cond_color.get(cond, "gray"),
            marker=PEER_MARKERS.get(peers, "o"),
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )

    from matplotlib.lines import Line2D

    cond_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cond_color[c],
            markeredgecolor="black",
            markersize=8,
            label=chunkdp_xtick_label(c),
        )
        for c in conditions
    ]
    peer_handles = [
        Line2D(
            [0],
            [0],
            marker=PEER_MARKERS.get(p, "o"),
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=8,
            label=f"{p} peers",
        )
        for p in sorted(summary["peers"].unique())
    ]
    ax.legend(
        handles=cond_handles + peer_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        fontsize=7.5,
        framealpha=0.95,
    )
    ax.set_xlabel(mean_accuracy_axis_label())
    ax.set_ylabel(mean_mia_auc_axis_label(auc_col))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    _save_fig(fig, out_path)
    plt.close()


def plot_lambda_by_condition(
    summary: pd.DataFrame,
    out_path: str,
    auc_col: str,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
) -> None:
    """Four panels (one per condition): x = peer count, grouped bars for S_λ."""
    summary = _condition_categorical(summary)
    conditions = [c for c in summary["condition"].cat.categories if (summary["condition"] == c).any()]
    peer_counts = sorted(summary["peers"].unique())
    n_peers = len(peer_counts)
    w = 0.22

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharey=True)
    axes_flat = axes.flatten()

    for ax, cond in zip(axes_flat, conditions):
        sub = summary[summary["condition"] == cond].set_index("peers").reindex(peer_counts)
        x = np.arange(n_peers)
        for i, lam in enumerate(lambdas):
            scores = deployment_score(sub["mean_accuracy"], sub["mean_auc"], lam)
            offset = (i - 1) * w
            ax.bar(
                x + offset,
                scores,
                width=w,
                label=rf"$\lambda={lam:g}$",
                edgecolor="black",
                linewidth=0.35,
            )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in peer_counts])
        ax.set_xlabel("Number of peers")
        ax.set_title(chunkdp_xtick_label(cond), fontsize=9)
        ax.grid(axis="y", alpha=0.25)

    axes_flat[0].set_ylabel(r"$S_\lambda = (1-\lambda)u - \lambda r$")
    axes_flat[2].set_ylabel(r"$S_\lambda = (1-\lambda)u - \lambda r$")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Tradeoff weight", loc="upper center", ncol=3, fontsize=9)
    fig.suptitle(f"Deployment scores by peer count ({auc_metric_title_suffix(auc_col)})", fontsize=10, y=1.02)
    plt.tight_layout()
    _save_fig(fig, out_path)
    plt.close()


def plot_score_vs_peers(
    summary: pd.DataFrame,
    out_path: str,
    lam: float,
) -> None:
    """Single λ score vs peer count, one line per condition."""
    summary = _condition_categorical(summary)
    peer_counts = sorted(summary["peers"].unique())
    conditions = summary["condition"].cat.categories.tolist()
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(conditions)))

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, cond in enumerate(conditions):
        sub = summary[summary["condition"] == cond].sort_values("peers")
        scores = deployment_score(sub["mean_accuracy"], sub["mean_auc"], lam)
        ax.plot(sub["peers"], scores, marker="o", linewidth=1.6, label=chunkdp_xtick_label(cond), color=cmap[i])

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Number of peers")
    ax.set_ylabel(rf"$S_\lambda$; $\lambda={lam:g}$")
    ax.set_xticks(peer_counts)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _save_fig(fig, out_path)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Plot peer-count mechanism sweep (2×2 ablation × peers).")
    p.add_argument(
        "--results-root",
        type=str,
        required=True,
        help="Directory containing peers_* subfolders (e.g. synced/results/peer_count_mechanism_sweep/er_p_0.08).",
    )
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for figures and CSVs.")
    p.add_argument("--round", type=int, default=None, help="Round to use (default: last round per file).")
    p.add_argument(
        "--auc-col",
        choices=["max_auc", "avg_auc"],
        default="max_auc",
        help="AUC column for privacy leakage.",
    )
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.5, help="λ for score line plot.")
    p.add_argument("--xmin", type=float, default=None)
    p.add_argument("--xmax", type=float, default=None)
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=None)
    args = p.parse_args()

    xmin = 0.0 if args.xmin is None else args.xmin
    xmax = 1.05 if args.xmax is None else args.xmax
    ymin = 0.0 if args.ymin is None else args.ymin
    ymax = 1.05 if args.ymax is None else args.ymax

    peer_dirs = discover_peer_dirs(args.results_root)
    print(f"Found peer directories: {[n for n, _ in peer_dirs]}")

    summary, _round_df = aggregate_sweep(peer_dirs, args.auc_col, args.round)
    summary = _condition_categorical(summary)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "peer_count_mechanism_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    lam_path = os.path.join(args.out_dir, "peer_count_mechanism_lambda_scores.csv")
    export_ablation_lambda_table(summary, lam_path)
    print(f"Saved lambda scores: {lam_path}")

    bar_path = os.path.join(args.out_dir, "peer_count_mechanism_bars.png")
    plot_grouped_bars(summary, bar_path, args.auc_col, ymin, ymax)
    print(f"Saved {bar_path} (+ .pdf)")

    line_path = os.path.join(args.out_dir, "peer_count_effect_lines.png")
    plot_peer_count_lines(summary, line_path, args.auc_col, ymin, ymax)
    print(f"Saved {line_path} (+ .pdf)")

    scatter_path = os.path.join(args.out_dir, "peer_count_mechanism_scatter.png")
    plot_scatter(summary, scatter_path, args.auc_col, xmin, xmax)
    print(f"Saved {scatter_path} (+ .pdf)")

    lambda_path = os.path.join(args.out_dir, "peer_count_mechanism_lambda_grouped.png")
    plot_lambda_by_condition(summary, lambda_path, args.auc_col)
    print(f"Saved {lambda_path} (+ .pdf)")

    score_path = os.path.join(args.out_dir, "peer_count_mechanism_score.png")
    plot_score_vs_peers(summary, score_path, float(np.clip(args.lambda_, 0.0, 1.0)))
    print(f"Saved {score_path} (+ .pdf)")


if __name__ == "__main__":
    main()
