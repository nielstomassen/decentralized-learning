#!/usr/bin/env python3
"""
Plot ChunkDP ablation with fixed-K chunking references (thesis hybrid ablation figures).

Produces the figure set under ``plots/hybrid_ablation_with_all_standard_k/er_p_<p>/``.

Inputs
------
Primary (``--results-dir``): 2×2 factorial ablation CSVs — baseline, DP only,
topology-aware chunking, ChunkDP (e.g. ``results/hybrid_ablation/er_p_<p>/``).

Additional (repeat ``--additional-results-dir``): fixed-K chunking runs, one directory
per K. Files with ``_cmstd_g<K>`` in the filename are labeled **Fixed-K chunking (K=…)**.
Typical roots: ``results/fixed_k_chunking_sweep/<K>_standard_chunking_sweep/er_p_<p>/``
for K ∈ {8, 16, 32, 64, 128}.

Outputs
-------
  - ``hybrid_ablation_bars.png`` — mean top-5 accuracy and mean maximum MIA AUC by condition
  - ``hybrid_privacy_utility_scatter.png`` — accuracy vs AUC (one point per condition)
  - ``hybrid_ablation_boxplots.png`` — node-level distributions with mean ± std overlay
  - ``hybrid_ablation_score.png`` — deployment score at ``--lambda`` (default 0.5)
  - ``hybrid_ablation_lambda_grouped.png`` — S_λ for λ ∈ {0.25, 0.5, 0.75}
  - ``hybrid_ablation_summary.csv``, ``hybrid_ablation_lambda_scores.csv``

Usage (er_p=0.08; repeat for 0.16 and each K)::

  python plotting/hybrid_ablation/hybrid_privacy_tradeoff.py \\
    --results-dir results/hybrid_ablation/er_p_0.08 \\
    --additional-results-dir synced/results/fixed_k_chunking_sweep/8_standard_chunking_sweep/er_p_0.08 \\
    --additional-results-dir synced/results/fixed_k_chunking_sweep/16_standard_chunking_sweep/er_p_0.08 \\
    --additional-results-dir synced/results/fixed_k_chunking_sweep/32_standard_chunking_sweep/er_p_0.08 \\
    --additional-results-dir synced/results/fixed_k_chunking_sweep/64_standard_chunking_sweep/er_p_0.08 \\
    --additional-results-dir synced/results/fixed_k_chunking_sweep/128_standard_chunking_sweep/er_p_0.08 \\
    --out-dir plots/hybrid_ablation_with_all_standard_k/er_p_0.08 \\
    --lambda 0.5 --auc-col max_auc
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from chunkdp_labels import (
    CHUNKDP_CONDITION,
    auc_metric_title_suffix,
    chunkdp_xtick_label,
    mean_accuracy_axis_label,
    mean_mia_auc_axis_label,
    node_accuracy_axis_label,
    node_mia_auc_axis_label,
    normalize_condition_label,
)
from hybrid_lambda_deployment_scores import (
    export_ablation_lambda_table,
    plot_ablation_grouped_three_lambdas,
)


def _save_fig(fig, path_png: str, dpi: int = 150) -> None:
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    base, _ = os.path.splitext(path_png)
    fig.savefig(base + ".pdf", bbox_inches="tight")

# Parse dp/chunk/noise from filename (e.g. ..._dp1_chunk0_noise0.5_...)
DP_REGEX = re.compile(r"_dp([01])(?:_|$)", re.IGNORECASE)
CHUNK_REGEX = re.compile(r"_chunk([01])(?:_|$)", re.IGNORECASE)
NOISE_REGEX = re.compile(r"_noise([0-9.]+)(?:_|$)", re.IGNORECASE)
# Fixed-K (_cmstd) chunking baseline runs tag filenames with _cmstd or _cmstd_g{K} (see mia_runner.save_csv).
CMSTD_REGEX = re.compile(r"_cmstd(?:_g\d+)?(?:_|$)", re.IGNORECASE)
CMSTD_K_REGEX = re.compile(r"_cmstd_g(\d+)(?:_|$)", re.IGNORECASE)

_FIXED_K_PLAIN = "Fixed-K chunking"
_FIXED_K_PREFIX = "Fixed-K chunking (K="


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
    base = os.path.basename(path)
    df["_is_standard_chunking"] = 1 if CMSTD_REGEX.search(base) else 0
    km = CMSTD_K_REGEX.search(base)
    df["_standard_chunk_k"] = int(km.group(1)) if km else 0
    df["_source_file"] = path
    return df


def condition_label(row) -> str:
    dp, ch = int(row["enable_dp"]), int(row["enable_chunking"])
    std = int(row.get("_is_standard_chunking", 0) or 0)
    std_k = int(row.get("_standard_chunk_k", 0) or 0)
    if dp and ch:
        return CHUNKDP_CONDITION
    if dp:
        return "DP only"
    if ch:
        if std:
            return f"{_FIXED_K_PREFIX}{std_k})" if std_k > 0 else _FIXED_K_PLAIN
        return "Topology-aware chunking"
    return "No DP, no chunk (baseline)"


def main():
    p = argparse.ArgumentParser(
        description=(
            "Plot ChunkDP ablation with fixed-K chunking references "
            "(hybrid_ablation_with_all_standard_k thesis figures)."
        ),
    )
    p.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help=(
            "Factorial ablation CSVs for one er_p (baseline, DP only, topology-aware chunking, ChunkDP), "
            "e.g. synced/results/final_hybrid_ablation/er_p_0.08"
        ),
    )
    p.add_argument(
        "--additional-results-dir",
        action="append",
        default=None,
        metavar="DIR",
        help=(
            "Fixed-K chunking CSV directory; repeat once per K (8, 16, 32, 64, 128). "
            "Runs are detected via _cmstd_gK in the filename."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory, e.g. synced/plots/hybrid_ablation_with_all_standard_k/er_p_0.08",
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
        help="Which AUC column to use for privacy leakage (default: max_auc).",
    )
    p.add_argument("--xmin", type=float, default=None, help="Scatter/bar accuracy axis lower limit.")
    p.add_argument("--xmax", type=float, default=None, help="Scatter/bar accuracy axis upper limit.")
    p.add_argument("--ymin", type=float, default=None, help="Scatter/bar accuracy axis lower limit (utility panel).")
    p.add_argument("--ymax", type=float, default=None, help="Scatter/bar accuracy axis upper limit (utility panel).")
    p.add_argument(
        "--xscale",
        type=str,
        choices=["linear", "log"],
        default="linear",
        help="Utility axis scale.",
    )
    p.add_argument(
        "--yscale",
        type=str,
        choices=["linear", "log"],
        default="linear",
        help="AUC axis scale.",
    )
    p.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=0.5,
        help="Tradeoff weight λ in score = (1−λ)u − λr for hybrid_ablation_score.png (default: 0.5).",
    )
    args = p.parse_args()
    additional_dirs = list(args.additional_results_dir or [])

    xmin = 0.0 if args.xmin is None else args.xmin
    xmax = 1.05 if args.xmax is None else args.xmax
    ymin = 0.0 if args.ymin is None else args.ymin
    ymax = 1.05 if args.ymax is None else args.ymax

    if not os.path.isdir(args.results_dir):
        raise FileNotFoundError(f"--results-dir is not a directory: {args.results_dir!r}")

    paths = [
        os.path.join(args.results_dir, f)
        for f in os.listdir(args.results_dir)
        if f.endswith(".csv")
    ]
    for d in additional_dirs:
        if not d or not os.path.isdir(d):
            print(f"Warning: skip missing results dir: {d}", file=sys.stderr)
            continue
        paths.extend(
            os.path.join(d, f)
            for f in os.listdir(d)
            if f.endswith(".csv")
        )
    paths = sorted(set(paths))

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
    for (src, enable_dp, enable_chunking, _std, _std_k), g in full.groupby(
        ["_source_file", "enable_dp", "enable_chunking", "_is_standard_chunking", "_standard_chunk_k"]
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
        std_acc_nodes = rdf["global_test_acc"].std()
        std_auc_nodes = rdf[args.auc_col].std()
        rows.append(
            {
                "enable_dp": enable_dp,
                "enable_chunking": enable_chunking,
                "condition": condition_label(rdf.iloc[0]),
                "mean_accuracy": acc,
                "mean_auc": auc,
                "std_acc_nodes": std_acc_nodes,
                "std_auc_nodes": std_auc_nodes,
                "n_nodes": len(rdf),
            }
        )

    by_cond = pd.DataFrame(rows).groupby(
        ["enable_dp", "enable_chunking", "condition"], as_index=False
    ).agg(
        mean_accuracy=("mean_accuracy", "mean"),
        std_accuracy_across_seeds=("mean_accuracy", "std"),
        mean_node_std_accuracy=("std_acc_nodes", "mean"),
        mean_auc=("mean_auc", "mean"),
        std_auc_across_seeds=("mean_auc", "std"),
        mean_node_std_auc=("std_auc_nodes", "mean"),
        n_seeds=("mean_accuracy", "count"),
    )

    # Fixed condition order for bars, scatter, and box plots (fixed-K Ks sorted ascending).
    std_conditions = sorted(
        [c for c in by_cond["condition"].unique() if str(c).startswith(_FIXED_K_PREFIX)],
        key=lambda s: int(re.search(r"K=(\d+)", str(s)).group(1)),
    )
    has_plain_std = (by_cond["condition"] == _FIXED_K_PLAIN).any()
    fixed_order = [
        "No DP, no chunk (baseline)",
        "DP only",
        "Topology-aware chunking",
        *([_FIXED_K_PLAIN] if has_plain_std else []),
        *std_conditions,
        CHUNKDP_CONDITION,
    ]
    order = list(fixed_order)
    lam = float(np.clip(getattr(args, "lambda_", 0.5), 0.0, 1.0))
    by_cond["privacy_risk"] = np.maximum(0, 2 * by_cond["mean_auc"] - 1)
    by_cond["score"] = (1.0 - lam) * by_cond["mean_accuracy"] - lam * by_cond["privacy_risk"]
    by_cond["condition"] = by_cond["condition"].map(normalize_condition_label)
    by_cond["condition"] = pd.Categorical(by_cond["condition"], categories=order, ordered=True)
    by_cond = by_cond.sort_values("condition").reset_index(drop=True)
    by_cond["condition"] = by_cond["condition"].map(normalize_condition_label)
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

    lam_tbl_path = os.path.join(args.out_dir, "hybrid_ablation_lambda_scores.csv")
    export_ablation_lambda_table(by_cond, lam_tbl_path)
    print(f"Saved multi-λ ablation scores: {lam_tbl_path}")
    plot_ablation_grouped_three_lambdas(
        by_cond,
        os.path.join(args.out_dir, "hybrid_ablation_lambda_grouped.png"),
        fixed_order,
        auc_metric_name=auc_metric_title_suffix(args.auc_col),
        show_title=False,
    )
    print("Saved hybrid_ablation_lambda_grouped.png (+ .pdf)")

    # Figure 1: Bar chart – accuracy and AUC by condition
    # Use short display labels and centered bars so category-to-bar mapping is clearer.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(by_cond))
    w = 0.62
    display_labels = [chunkdp_xtick_label(c) for c in by_cond["condition"].astype(str).tolist()]

    acc_vals = by_cond["mean_accuracy"].values
    acc_err = by_cond["mean_node_std_accuracy"].fillna(0).values
    ax1.bar(x, acc_vals, w, yerr=acc_err, capsize=3, color="tab:blue")
    ax1.set_ylabel(mean_accuracy_axis_label())
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_labels, rotation=25, ha="right", fontsize=8)
    ax1.set_ylim(ymin, ymax)
    ax1.set_yscale(args.yscale)
    ax1.grid(axis="y", alpha=0.22)

    auc_vals = by_cond["mean_auc"].values
    auc_err = by_cond["mean_node_std_auc"].fillna(0).values
    ax2.bar(x, auc_vals, w, yerr=auc_err, capsize=3, color="tab:red")
    ax2.set_ylabel(mean_mia_auc_axis_label(args.auc_col))
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_labels, rotation=25, ha="right", fontsize=8)
    ax2.set_ylim(0.4, 1)
    ax2.set_yscale(args.yscale)
    ax2.grid(axis="y", alpha=0.22)

    plt.tight_layout()
    bar_path = os.path.join(args.out_dir, "hybrid_ablation_bars.png")
    _save_fig(fig, bar_path)
    plt.close()
    print(f"Saved {bar_path} (+ .pdf)")

    # Figure 2: Scatter – accuracy vs AUC (one point per condition); lower AUC = more private
    fig, ax = plt.subplots(figsize=(8.2, 5))
    for _, row in by_cond.iterrows():
        ax.scatter(
            row["mean_accuracy"],
            row["mean_auc"],
            s=120,
            label=chunkdp_xtick_label(row["condition"]),
        )
    ax.set_xlabel(mean_accuracy_axis_label())
    ax.set_ylabel(mean_mia_auc_axis_label(args.auc_col))
    # Keep legend below the axes to avoid covering points.
    ncol_legend = max(2, min(4, len(by_cond)))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=ncol_legend,
        fontsize=8,
        framealpha=0.95,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.4, 1)
    ax.set_xscale(args.xscale)
    ax.set_yscale(args.yscale)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    scatter_path = os.path.join(args.out_dir, "hybrid_privacy_utility_scatter.png")
    _save_fig(fig, scatter_path)
    plt.close()
    print(f"Saved {scatter_path} (+ .pdf)")

    # Figure 3: Box plots – fixed order; boxes + mean ± std overlay
    cond_order = [c for c in fixed_order if (round_df["condition"] == c).any()]
    if not cond_order:
        print("Warning: no data for box plots; skip hybrid ablation boxplots.", file=sys.stderr)
    else:
        acc_by_cond = [
            round_df.loc[round_df["condition"] == c, "_acc"].dropna().values for c in cond_order
        ]
        auc_by_cond = [
            round_df.loc[round_df["condition"] == c, "_auc"].dropna().values for c in cond_order
        ]
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
            s_a = row["mean_node_std_accuracy"] if pd.notna(row["mean_node_std_accuracy"]) else 0
            mu_u = row["mean_auc"]
            s_u = row["mean_node_std_auc"] if pd.notna(row["mean_node_std_auc"]) else 0
            bx1.plot([i], [mu_a], "k*", markersize=10)
            bx1.errorbar([i], [mu_a], yerr=[s_a], fmt="none", color="black", capsize=3)
            bx2.plot([i], [mu_u], "k*", markersize=10)
            bx2.errorbar([i], [mu_u], yerr=[s_u], fmt="none", color="black", capsize=3)
            if len(acc_by_cond[i]) > 0:
                bx1.plot(i, np.max(acc_by_cond[i]), "o", color="tab:blue", markersize=5, alpha=0.8, zorder=5)
            if len(auc_by_cond[i]) > 0:
                bx2.plot(i, np.max(auc_by_cond[i]), "o", color="tab:red", markersize=5, alpha=0.8, zorder=5)
        bx1.set_xticks(pos)
        bx1.set_xticklabels([chunkdp_xtick_label(c) for c in cond_order], rotation=15, ha="right")
        bx1.set_ylabel(node_accuracy_axis_label())
        bx1.set_ylim(ymin, ymax)
        bx2.set_xticks(pos)
        bx2.set_xticklabels([chunkdp_xtick_label(c) for c in cond_order], rotation=15, ha="right")
        bx2.set_ylabel(node_mia_auc_axis_label(args.auc_col))
        bx2.set_ylim(0.4, 1)
        plt.tight_layout()
        box_path = os.path.join(args.out_dir, "hybrid_ablation_boxplots.png")
        _save_fig(fig3, box_path)
        plt.close()
        print(f"Saved {box_path} (+ .pdf)")

    # Figure 4: Score bar chart – conditions ordered by score (best first)
    by_cond_score = by_cond.sort_values("score", ascending=False).reset_index(drop=True)
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(by_cond_score))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(by_cond_score))[::-1])
    ax4.bar(x, by_cond_score["score"].values, color=colors)
    ax4.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax4.set_ylabel(f"Score ((1−λ)u − λr); λ = {lam}")
    ax4.set_xticks(x)
    ax4.set_xticklabels(
        [chunkdp_xtick_label(c) for c in by_cond_score["condition"].values],
        rotation=15,
        ha="right",
    )
    plt.tight_layout()
    score_path = os.path.join(args.out_dir, "hybrid_ablation_score.png")
    _save_fig(fig4, score_path)
    plt.close()
    print(f"Saved {score_path} (+ .pdf)")


if __name__ == "__main__":
    main()
