#!/usr/bin/env python3
"""Compare topology-aware chunking and hybrid: normal hybrid ablation vs broadcastSame.
Used for Appendix experiment in the thesis
Same per-seed aggregation as ``hybrid_privacy_tradeoff`` (last round, mean over nodes,
then mean ± std across seeds). Excludes ``_cmstd`` files from the normal ablation directory.

Example:
  python3 -m plotting.appendix.normal_vs_broadcast_same_ta_hybrid \
    --factorial-root results/hybrid_ablation \
    --broadcast-root results/appendix/flatBroadcastSame \
    --out-path plots/appendix/chunk_variant/normal_vs_broadcast_same_ta_hybrid.png

Writes ``<stem>_comparison.csv`` next to the figure (means, SEs, broadcast−normal differences).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plotting.hybrid_ablation.chunkdp_labels import (  # noqa: E402
    CHUNKDP_CONDITION,
    mean_accuracy_axis_label,
    mean_mia_auc_axis_label,
)

NORMAL_STYLE_LABEL = "Normal"
TA_CONDITION = "Topology-aware chunking"
from plotting.hybrid_ablation.hybrid_privacy_tradeoff import (  # noqa: E402
    CMSTD_REGEX,
    CHUNK_REGEX,
    DP_REGEX,
    condition_label,
    load_and_label,
)


def _ta_hybrid_csv_paths(results_dir: Path) -> list[str]:
    """Non-fixed-K dp0_chunk1 and dp1_chunk1 CSV paths under ``results_dir``."""
    out: list[str] = []
    if not results_dir.is_dir():
        return out
    for p in sorted(results_dir.glob("*.csv")):
        base = p.name
        if CMSTD_REGEX.search(base):
            continue
        md = DP_REGEX.search(base)
        mc = CHUNK_REGEX.search(base)
        if not md or not mc:
            continue
        dp, ch = int(md.group(1)), int(mc.group(1))
        if (dp, ch) in ((0, 1), (1, 1)):
            out.append(str(p))
    return out


def aggregate_ta_hybrid(
    paths: list[str],
    auc_col: str,
    target_round: int | None,
) -> pd.DataFrame:
    """Return one row per condition (Topology-aware chunking, ChunkDP)."""
    if not paths:
        return pd.DataFrame()
    frames = []
    for path in paths:
        try:
            frames.append(load_and_label(path))
        except Exception as e:
            print(f"Warning: skip {path}: {e}", file=sys.stderr)
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    full["condition"] = full.apply(condition_label, axis=1)
    want = {TA_CONDITION, CHUNKDP_CONDITION}
    rows = []
    for (src, enable_dp, enable_chunking, _std, _std_k), g in full.groupby(
        ["_source_file", "enable_dp", "enable_chunking", "_is_standard_chunking", "_standard_chunk_k"]
    ):
        if target_round is not None:
            rdf = g[g["round"] == target_round]
        else:
            lr = g["round"].max()
            rdf = g[g["round"] == lr]
        if rdf.empty:
            continue
        cond = condition_label(rdf.iloc[0])
        if cond not in want:
            continue
        rows.append(
            {
                "condition": cond,
                "mean_accuracy": rdf["global_test_acc"].mean(),
                "mean_auc": rdf[auc_col].mean(),
                "std_acc_nodes": rdf["global_test_acc"].std(),
                "std_auc_nodes": rdf[auc_col].std(),
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .groupby("condition", as_index=False)
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


def _save_fig(fig: plt.Figure, path_png: str, dpi: int = 150) -> None:
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    base, _ = os.path.splitext(path_png)
    fig.savefig(base + ".pdf", bbox_inches="tight")


def plot_compare(
    out_path: str,
    factorial_by_p: dict[str, pd.DataFrame],
    broadcast_by_p: dict[str, pd.DataFrame],
    auc_col: str,
    variant_style_label: str,
) -> None:
    er_ps = sorted(factorial_by_p.keys(), key=float)
    fig, axes = plt.subplots(len(er_ps), 2, figsize=(9.5, 4.2 * len(er_ps)), squeeze=False)
    cond_order = [TA_CONDITION, CHUNKDP_CONDITION]
    x_labels = ["Topology-aware\nchunking", CHUNKDP_CONDITION]
    width = 0.35
    colors = {"factorial": "#1f77b4", "broadcast": "#ff7f0e"}

    for row, p in enumerate(er_ps):
        fac = factorial_by_p[p].set_index("condition")
        br = broadcast_by_p[p].set_index("condition")
        for col, (metric, ylabel, err_within) in enumerate(
            [
                ("mean_accuracy", mean_accuracy_axis_label(), "mean_node_std_accuracy"),
                ("mean_auc", mean_mia_auc_axis_label(auc_col), "mean_node_std_auc"),
            ]
        ):
            ax = axes[row, col]
            x = np.arange(len(cond_order))
            for i, (label, series_name) in enumerate(
                [
                    (NORMAL_STYLE_LABEL, "factorial"),
                    (variant_style_label, "broadcast"),
                ]
            ):
                tbl = fac if series_name == "factorial" else br
                vals = []
                err = []
                for c in cond_order:
                    if c not in tbl.index:
                        vals.append(np.nan)
                        err.append(0.0)
                    else:
                        r = tbl.loc[c]
                        vals.append(float(r[metric]))
                        e = r[err_within] if pd.notna(r[err_within]) else 0.0
                        err.append(float(e))
                offset = width * (i - 0.5)
                ax.bar(
                    x + offset,
                    vals,
                    width,
                    yerr=np.array(err),
                    capsize=3,
                    label=label,
                    color=colors["factorial" if series_name == "factorial" else "broadcast"],
                    alpha=0.88,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_ylabel(ylabel)
            ax.set_title(f"ER p = {p}")
            ax.grid(axis="y", alpha=0.25)
            if col == 1:
                ax.set_ylim(0.4, 1.0)
            else:
                ax.set_ylim(0.0, 1.05)
            ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    _save_fig(fig, out_path)
    plt.close(fig)
    print(f"Saved {out_path} (+ .pdf)")


def plot_scatter_panel(
    out_path: str,
    factorial_by_p: dict[str, pd.DataFrame],
    broadcast_by_p: dict[str, pd.DataFrame],
    auc_col: str,
    variant_style_label: str,
) -> None:
    """Second figure: accuracy vs AUC scatter, one panel per er_p."""
    er_ps = sorted(factorial_by_p.keys(), key=float)
    fig, axes = plt.subplots(1, len(er_ps), figsize=(5.2 * len(er_ps), 4.8), squeeze=False)
    axes = axes.ravel()
    for ax, p in zip(axes, er_ps):
        fac = factorial_by_p[p].set_index("condition")
        br = broadcast_by_p[p].set_index("condition")
        series = [
            (fac, TA_CONDITION, "o", "#1f77b4", f"{TA_CONDITION} ({NORMAL_STYLE_LABEL})"),
            (fac, CHUNKDP_CONDITION, "s", "#1f77b4", f"{CHUNKDP_CONDITION} ({NORMAL_STYLE_LABEL})"),
            (br, TA_CONDITION, "o", "#ff7f0e", f"{TA_CONDITION} ({variant_style_label})"),
            (br, CHUNKDP_CONDITION, "s", "#ff7f0e", f"{CHUNKDP_CONDITION} ({variant_style_label})"),
        ]
        for tbl, cond, mk, color, leg in series:
            if cond not in tbl.index:
                continue
            r = tbl.loc[cond]
            ax.scatter(
                r["mean_accuracy"],
                r["mean_auc"],
                s=130,
                marker=mk,
                c=color,
                edgecolors="black",
                linewidths=0.6,
                zorder=3,
                label=leg,
            )
        ax.set_xlabel(mean_accuracy_axis_label())
        ax.set_ylabel(mean_mia_auc_axis_label(auc_col))
        ax.set_title(f"ER p = {p}")
        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    base, ext = os.path.splitext(out_path)
    scatter_path = f"{base}_scatter{ext or '.png'}"
    os.makedirs(os.path.dirname(scatter_path) or ".", exist_ok=True)
    _save_fig(fig, scatter_path)
    plt.close(fig)
    print(f"Saved {scatter_path} (+ .pdf)")


def export_comparison_csv(
    out_path: str,
    factorial_by_p: dict[str, pd.DataFrame],
    broadcast_by_p: dict[str, pd.DataFrame],
    auc_col: str,
) -> str:
    """Write CSV with aligned metrics and broadcastSame − Normal differences."""
    cond_order = [TA_CONDITION, CHUNKDP_CONDITION]
    rows: list[dict] = []
    for p in sorted(factorial_by_p.keys(), key=float):
        fac = factorial_by_p[p].set_index("condition")
        br = broadcast_by_p[p].set_index("condition")
        for cond in cond_order:
            if cond not in fac.index or cond not in br.index:
                raise ValueError(f"Missing condition {cond!r} for er_p={p}")
            fn, bn = fac.loc[cond], br.loc[cond]
            rows.append(
                {
                    "er_p": float(p),
                    "condition": cond,
                    "mia_auc_column": auc_col,
                    "normal_mean_test_accuracy": fn["mean_accuracy"],
                    "broadcast_mean_test_accuracy": bn["mean_accuracy"],
                    "diff_mean_test_accuracy_broadcast_minus_normal": float(bn["mean_accuracy"] - fn["mean_accuracy"]),
                    "normal_mean_mia_auc": fn["mean_auc"],
                    "broadcast_mean_mia_auc": bn["mean_auc"],
                    "diff_mean_mia_auc_broadcast_minus_normal": float(bn["mean_auc"] - fn["mean_auc"]),
                    "normal_n_seeds": int(fn["n_seeds"]),
                    "broadcast_n_seeds": int(bn["n_seeds"]),
                    "normal_std_test_accuracy_across_seeds": fn["std_accuracy_across_seeds"],
                    "broadcast_std_test_accuracy_across_seeds": bn["std_accuracy_across_seeds"],
                    "normal_mean_within_run_std_test_accuracy_over_nodes": fn["mean_node_std_accuracy"],
                    "broadcast_mean_within_run_std_test_accuracy_over_nodes": bn["mean_node_std_accuracy"],
                    "normal_std_mia_auc_across_seeds": fn["std_auc_across_seeds"],
                    "broadcast_std_mia_auc_across_seeds": bn["std_auc_across_seeds"],
                    "normal_mean_within_run_std_mia_auc_over_nodes": fn["mean_node_std_auc"],
                    "broadcast_mean_within_run_std_mia_auc_over_nodes": bn["mean_node_std_auc"],
                }
            )
    out_csv = f"{os.path.splitext(out_path)[0]}_comparison.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--factorial-root",
        type=str,
        default="synced/results/final_hybrid_ablation",
        help="Root containing er_p_<p> normal hybrid-ablation CSVs.",
    )
    ap.add_argument(
        "--broadcast-root",
        type=str,
        default="synced/results/broadcastSame",
        help="Root containing er_p_<p> broadcastSame CSVs.",
    )
    ap.add_argument(
        "--er-ps",
        type=str,
        default="0.08,0.16",
        help="Comma-separated ER p values (must have matching er_p_* subdirs).",
    )
    ap.add_argument(
        "--out-path",
        type=str,
        default="synced/plots/broadcastSame/normal_vs_broadcast_same_ta_hybrid.png",
        help="Output PNG path (.pdf written alongside). Scatter uses *_scatter.png; table uses *_comparison.csv.",
    )
    ap.add_argument("--auc-col", type=str, choices=["max_auc", "avg_auc"], default="max_auc")
    ap.add_argument("--round", type=int, default=None, help="Round index (default: last per file).")
    ap.add_argument(
        "--variant-style-label",
        type=str,
        default="Fixed-K style",
        help="Legend suffix for the non-factorial arm, e.g. 'Fixed-K style' or 'broadcast same'.",
    )
    args = ap.parse_args()

    factorial_root = Path(args.factorial_root)
    broadcast_root = Path(args.broadcast_root)
    er_ps = [x.strip() for x in args.er_ps.split(",") if x.strip()]

    factorial_by_p: dict[str, pd.DataFrame] = {}
    broadcast_by_p: dict[str, pd.DataFrame] = {}
    for p in er_ps:
        fdir = factorial_root / f"er_p_{p}"
        bdir = broadcast_root / f"er_p_{p}"
        fp = _ta_hybrid_csv_paths(fdir)
        bp = _ta_hybrid_csv_paths(bdir)
        factorial_by_p[p] = aggregate_ta_hybrid(fp, args.auc_col, args.round)
        broadcast_by_p[p] = aggregate_ta_hybrid(bp, args.auc_col, args.round)
        for name, df in [("normal ablation", factorial_by_p[p]), ("broadcastSame", broadcast_by_p[p])]:
            if df.empty:
                raise SystemExit(f"No TA/hybrid data for er_p={p} under {name} ({fdir if name == 'normal ablation' else bdir})")

    plot_compare(
        args.out_path,
        factorial_by_p,
        broadcast_by_p,
        args.auc_col,
        args.variant_style_label,
    )
    plot_scatter_panel(
        args.out_path,
        factorial_by_p,
        broadcast_by_p,
        args.auc_col,
        args.variant_style_label,
    )
    export_comparison_csv(args.out_path, factorial_by_p, broadcast_by_p, args.auc_col)


if __name__ == "__main__":
    main()
