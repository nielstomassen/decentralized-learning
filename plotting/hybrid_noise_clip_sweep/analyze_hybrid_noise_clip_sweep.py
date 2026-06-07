#!/usr/bin/env python3
"""Analyze ChunkDP (σ, C) noise×clip sweep for thesis figures.

Produces plots under ``plots/noise_sweep_with_standard_k_refs/er_p_<p>/``
(scatter with ablation + fixed-K reference diamonds, λ panels, bars, score CSVs).

Inputs (per ER edge probability *p*)
------------------------------------
Primary (``--results-glob``): ChunkDP sweep CSVs only (``dp1_chunk1``), e.g.
``results/hybrid_nosie_clip_sweep/er_p_<p>/*.csv``.

Ablation references (``--ablation-ref-dir``): factorial ablation without the ChunkDP arm —
baseline, DP only, topology-aware chunk only — e.g.
``results/hybrid_ablation/er_p_<p>/``.

Fixed-K references (``--er-p``): **no separate CLI paths**. Passing ``--er-p 0.08`` or
``--er-p 0.16`` loads K ∈ {8, 16, 32, 64, 128} from ``FIXED_K_REF_DIRS`` below

Outputs
-------
  - ``hybrid_noise_clip_sweep_summary.csv``
  - ``hybrid_noise_clip_sweep_lambda_scores.csv``, ``*_lambda_optima_meta.csv``
  - ``hybrid_noise_clip_sweep_lambda_optima_grouped.{png,pdf}``
  - ``hybrid_noise_clip_sweep_lambda_panels.{png,pdf}``
  - ``hybrid_noise_clip_sweep_scatter.{png,pdf}`` (full context + zoom)
  - ``hybrid_noise_clip_sweep_bars.{png,pdf}``, ``hybrid_noise_clip_sweep_score.{png,pdf}``

Usage::

  python3 -m plotting.hybrid_noise_clip_sweep.analyze_hybrid_noise_clip_sweep \
    --results-glob 'results/hybrid_noise_clip_sweep/er_p_0.08/*.csv' \
    --out-dir plots/hybrid_noise_clip_sweep/er_p_0.08 \
    --ablation-ref-dir results/hybrid_ablation/er_p_0.08 \
    --er-p 0.08 --lambda 0.5 --auc-col max_auc
"""

from __future__ import annotations

import argparse
import glob as glob_module
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Fixed-K chunking reference directories (scatter diamonds + λ-panel refs).
# Selected by --er-p; edit here if result roots move.
FIXED_K_REF_DIRS: dict[str, list[tuple[str, str]]] = {
    "0.08": [
        ("Fixed-K chunking K=8", "results/fixed_k_chunking_sweep/8_standard_chunking_sweep/er_p_0.08"),
        ("Fixed-K chunking K=16", "results/fixed_k_chunking_sweep/16_standard_chunking_sweep/er_p_0.08"),
        ("Fixed-K chunking K=32", "results/fixed_k_chunking_sweep/32_standard_chunking_sweep/er_p_0.08"),
        ("Fixed-K chunking K=64", "results/fixed_k_chunking_sweep/64_standard_chunking_sweep/er_p_0.08"),
        ("Fixed-K chunking K=128", "results/fixed_k_chunking_sweep/128_standard_chunking_sweep/er_p_0.08"),
    ],
    "0.16": [
        ("Fixed-K chunking K=8", "results/fixed_k_chunking_sweep/8_standard_chunking_sweep/er_p_0.16"),
        ("Fixed-K chunking K=16", "results/fixed_k_chunking_sweep/16_standard_chunking_sweep/er_p_0.16"),
        ("Fixed-K chunking K=32", "results/fixed_k_chunking_sweep/32_standard_chunking_sweep/er_p_0.16"),
        ("Fixed-K chunking K=64", "results/fixed_k_chunking_sweep/64_standard_chunking_sweep/er_p_0.16"),
        ("Fixed-K chunking K=128", "results/fixed_k_chunking_sweep/128_standard_chunking_sweep/er_p_0.16"),
    ],
}

from plotting.hybrid_ablation.chunkdp_labels import (
    auc_metric_title_suffix,
    mean_accuracy_axis_label,
    mean_mia_auc_axis_label,
)
from plotting.hybrid_ablation.hybrid_lambda_deployment_scores import (
    export_sweep_lambda_optima_meta,
    export_sweep_lambda_table,
    plot_noise_sweep_three_lambdas,
    plot_sweep_optima_grouped_three_lambdas,
)
from plotting.hybrid_ablation.hybrid_privacy_tradeoff import load_and_label as load_ablation_csv


def _save_fig(fig, path_png: str, dpi: int = 150) -> None:
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    base, _ = os.path.splitext(path_png)
    fig.savefig(base + ".pdf", bbox_inches="tight")


def fixed_k_ref_specs(er_p: str) -> list[str]:
    """Return ``--extra-scatter-ref`` specs (LABEL::DIR) for the given ER edge probability."""
    key = str(er_p)
    if key not in FIXED_K_REF_DIRS:
        raise ValueError(f"--er-p must be one of {sorted(FIXED_K_REF_DIRS)}; got {er_p!r}")
    out: list[str] = []
    for label, rel_dir in FIXED_K_REF_DIRS[key]:
        out.append(f"{label}::{(_REPO_ROOT / rel_dir).as_posix()}")
    return out


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


def load_ablation_reference_points(
    ablation_dir: str,
    auc_col: str,
    target_round: int | None,
) -> list[dict]:
    """Network means (avg over nodes at final round) for non-hybrid ablation CSVs; averaged over seeds."""
    paths = sorted(
        os.path.join(ablation_dir, f)
        for f in os.listdir(ablation_dir)
        if f.endswith(".csv")
    )
    by_key: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for path in paths:
        df = load_ablation_csv(path)
        dp = int(df["enable_dp"].iloc[0])
        ch = int(df["enable_chunking"].iloc[0])
        if dp == 1 and ch == 1:
            continue
        if target_round is not None:
            rdf = df[df["round"] == target_round]
        else:
            lr = df["round"].max()
            rdf = df[df["round"] == lr]
        if rdf.empty:
            continue
        acc = float(rdf["global_test_acc"].mean())
        auc = float(rdf[auc_col].mean())
        by_key.setdefault((dp, ch), []).append((acc, auc))

    label_style = {
        (0, 0): ("Baseline", "#2ca02c"),
        (0, 1): ("Topology-aware chunk only", "#ff7f0e"),
        (1, 0): ("DP only", "#9467bd"),
    }
    out: list[dict] = []
    for key, pairs in by_key.items():
        if key not in label_style:
            continue
        lbl, col = label_style[key]
        out.append(
            {
                "label": lbl,
                "acc": float(np.mean([p[0] for p in pairs])),
                "auc": float(np.mean([p[1] for p in pairs])),
                "color": col,
                "is_standard": False,
            }
        )
    return out


def load_directory_mean_reference_point(
    ref_dir: str,
    auc_col: str,
    target_round: int | None,
) -> tuple[float, float] | None:
    """
    Mean test accuracy and mean MIA AUC over all CSV seeds in ``ref_dir`` (final round per file;
    each file: mean over nodes). Used for fixed-K chunking sweep folders on the noise-sweep scatter.
    """
    if not os.path.isdir(ref_dir):
        return None
    paths = sorted(
        os.path.join(ref_dir, f)
        for f in os.listdir(ref_dir)
        if f.endswith(".csv")
    )
    pairs: list[tuple[float, float]] = []
    for path in paths:
        try:
            df = load_ablation_csv(path)
        except Exception:
            continue
        if target_round is not None:
            rdf = df[df["round"] == target_round]
        else:
            lr = df["round"].max()
            rdf = df[df["round"] == lr]
        if rdf.empty:
            continue
        acc = float(rdf["global_test_acc"].mean())
        auc = float(rdf[auc_col].mean())
        pairs.append((acc, auc))
    if not pairs:
        return None
    return float(np.mean([p[0] for p in pairs])), float(np.mean([p[1] for p in pairs]))


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
        description=(
            "Plot ChunkDP noise×clip sweep with ablation and fixed-K reference overlays "
            "(noise_sweep_er008_with_standard_k_refs thesis figures)."
        ),
    )
    p.add_argument(
        "--results-glob",
        type=str,
        required=True,
        help=(
            "Glob for ChunkDP sweep CSVs (dp1_chunk1 only), e.g. "
            "synced/results/noise_sweep_multiseeds/er_p_0.08/*.csv"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory, e.g. synced/plots/noise_sweep_er008_with_standard_k_refs/er_p_0.08",
    )
    p.add_argument(
        "--ablation-ref-dir",
        type=str,
        required=True,
        help=(
            "Factorial ablation CSVs for reference diamonds (baseline, DP only, topology-aware chunking), "
            "e.g. synced/results/final_hybrid_ablation/er_p_0.08"
        ),
    )
    p.add_argument(
        "--er-p",
        type=str,
        choices=sorted(FIXED_K_REF_DIRS),
        required=True,
        help=(
            "ER edge probability p. Selects fixed-K chunking reference dirs (K=8,16,32,64,128) "
            "from FIXED_K_REF_DIRS in this file — no need to pass each K directory on the CLI."
        ),
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
    p.add_argument(
        "--no-score-plot",
        action="store_true",
        help="Skip score bar chart (configs ordered by score, higher = better).",
    )
    p.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=0.5,
        help="Tradeoff weight λ in score = (1−λ)u − λr, r = max(0, 2·AUC−1). Default 0.5; clipped to [0,1].",
    )
    p.add_argument(
        "--extra-scatter-ref",
        action="append",
        default=None,
        metavar="LABEL::DIR",
        help=(
            "Optional extra reference diamond (LABEL::DIR), in addition to the fixed-K dirs "
            "loaded via --er-p. Repeatable."
        ),
    )
    p.add_argument(
        "--scatter-inset-frac",
        type=float,
        default=0.14,
        help="Inset half-margin as a fraction of the hybrid span on each axis (default 0.14).",
    )
    args = p.parse_args()
    extra_scatter_specs = fixed_k_ref_specs(args.er_p)
    if args.extra_scatter_ref:
        extra_scatter_specs.extend(args.extra_scatter_ref)

    paths = sorted(glob_module.glob(args.results_glob))
    if not paths:
        raise FileNotFoundError(f"No CSV files match --results-glob {args.results_glob!r}")

    ablation_dir = args.ablation_ref_dir
    if not os.path.isdir(ablation_dir):
        raise FileNotFoundError(f"--ablation-ref-dir is not a directory: {ablation_dir!r}")

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

    # Keep node-level rows for target round only (one row per node per run)
    if target_round is not None:
        round_df = full[full["round"] == target_round].copy()
    else:
        last_round = full.groupby("_source_file")["round"].transform("max")
        round_df = full[full["round"] == last_round].copy()
    round_df["_acc"] = round_df["global_test_acc"]
    round_df["_auc"] = round_df[args.auc_col]

    # Per run: mean over nodes; then across runs (seeds): mean ± std of run-level metrics
    run_level = round_df.groupby(
        ["_source_file", "dp_noise", "dp_max_grad_norm"], as_index=False
    ).agg(
        mean_accuracy=("_acc", "mean"),
        mean_auc=("_auc", "mean"),
        n_nodes=("_acc", "count"),
    )
    by_config = run_level.groupby(["dp_noise", "dp_max_grad_norm"], as_index=False).agg(
        mean_accuracy=("mean_accuracy", "mean"),
        std_accuracy=("mean_accuracy", "std"),
        mean_auc=("mean_auc", "mean"),
        std_auc=("mean_auc", "std"),
        n_runs=("mean_accuracy", "count"),
        n_nodes=("n_nodes", "first"),
    )
    by_config = by_config.reset_index(drop=True)
    by_config["privacy_risk"] = np.maximum(0, 2 * by_config["mean_auc"] - 1)
    os.makedirs(args.out_dir, exist_ok=True)
    lam_csv = os.path.join(args.out_dir, "hybrid_noise_clip_sweep_lambda_scores.csv")
    export_sweep_lambda_table(by_config, lam_csv)
    print(f"Saved multi-λ scores: {lam_csv}")

    opt_csv = os.path.join(args.out_dir, "hybrid_noise_clip_sweep_lambda_optima_meta.csv")
    export_sweep_lambda_optima_meta(by_config, opt_csv)
    print(f"Saved λ-optima metadata: {opt_csv}")
    plot_sweep_optima_grouped_three_lambdas(
        by_config,
        os.path.join(args.out_dir, "hybrid_noise_clip_sweep_lambda_optima_grouped.png"),
        auc_metric_name=auc_metric_title_suffix(args.auc_col),
        show_title=False,
    )
    print("Saved hybrid_noise_clip_sweep_lambda_optima_grouped.png (+ .pdf)")

    lam = float(np.clip(args.lambda_, 0.0, 1.0))
    by_config["score"] = (1.0 - lam) * by_config["mean_accuracy"] - lam * by_config["privacy_risk"]
    by_config = by_config.sort_values("score", ascending=False).reset_index(drop=True)

    os.makedirs(args.out_dir, exist_ok=True)

    # Summary CSV (includes score; rows ordered by score descending = best first)
    summary_path = os.path.join(args.out_dir, "hybrid_noise_clip_sweep_summary.csv")
    by_config.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
    print(by_config.to_string(index=False))

    # Scatter: left = full context (hybrid + ablation refs); right = zoom on hybrid cluster only.
    by_plot = by_config.sort_values(["dp_noise", "dp_max_grad_norm"]).reset_index(drop=True)
    # High-contrast palette (colorblind-friendlier than tab10/tab20 for dense legends).
    # Deliberately minimizes adjacent green-like tones so configs are easier to separate.
    _distinct_palette = [
        "#1f77b4",  # blue
        "#d62728",  # red
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#17becf",  # cyan
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#bcbd22",  # olive
        "#2ca02c",  # green
        "#7f7f7f",  # gray
        "#003f5c",  # deep blue
        "#ef5675",  # rose
        "#ffa600",  # amber
        "#7a5195",  # violet
        "#2f4b7c",  # navy
        "#f95d6a",  # coral red
        "#a05195",  # magenta purple
        "#665191",  # indigo
        "#00876c",  # teal green
        "#ed553b",  # warm red-orange
        "#4cc9f0",  # sky cyan
        "#ff006e",  # hot pink
        "#3a86ff",  # bright blue
        "#fb5607",  # vivid orange
    ]
    n_hybrid = len(by_plot)
    if n_hybrid <= len(_distinct_palette):
        hybrid_colors = _distinct_palette[:n_hybrid]
    else:
        _hsv = plt.colormaps["hsv"]
        hybrid_colors = list(_distinct_palette)
        hybrid_colors.extend(_hsv(i / max(n_hybrid - len(_distinct_palette), 1)) for i in range(n_hybrid - len(_distinct_palette)))

    ref_points: list[dict] = []
    ref_points = load_ablation_reference_points(
        ablation_dir, args.auc_col, target_round
    )
    if ref_points:
        print(f"Scatter ablation overlays from: {ablation_dir} ({len(ref_points)} refs)")
    else:
        print(f"Warning: no ablation reference points found in {ablation_dir}")

    # Optional extra reference diamonds (e.g. fixed-K chunking sweeps at different global K)
    # Blue ramp: visually distinct from ablation Baseline (#2ca02c green), DP (#9467bd), chunk (#ff7f0e).
    _fixed_k_scatter_colors = [
        "#9ecae1",
        "#6baed6",
        "#3182bd",
        "#08519c",
        "#042044",
    ]
    for i, spec in enumerate(extra_scatter_specs):
        if "::" not in spec:
            raise ValueError(
                f"--extra-scatter-ref must be 'LABEL::DIR' (two colons); got: {spec!r}"
            )
        label, ref_dir = spec.split("::", 1)
        label, ref_dir = label.strip(), ref_dir.strip()
        pt = load_directory_mean_reference_point(ref_dir, args.auc_col, target_round)
        if pt is None:
            print(f"Warning: no usable CSVs for extra scatter ref {label!r} in {ref_dir}")
            continue
        acc, auc = pt
        ref_points.append(
            {
                "label": label,
                "acc": acc,
                "auc": auc,
                "color": _fixed_k_scatter_colors[i % len(_fixed_k_scatter_colors)],
                "is_standard": True,
            }
        )
        print(f"Extra scatter ref {label!r}: acc={acc:.4f}, auc={auc:.4f} (from {ref_dir})")

    # Lambda panels: include non-hybrid reference scores (DP/chunk + fixed-K chunking Ks; baseline omitted upstream).
    _fixed_k_label = lambda s: str(s).startswith("Fixed-K chunking") or str(s).startswith("Standard chunking")
    panel_ref_rows: list[dict] = []
    for ref in ref_points:
        lbl = str(ref.get("label", ""))
        if lbl in {"DP only", "Topology-aware chunk only"} or _fixed_k_label(lbl):
            panel_ref_rows.append(
                {
                    "label": lbl,
                    "mean_accuracy": float(ref["acc"]),
                    "mean_auc": float(ref["auc"]),
                }
            )
    _panel_order = {"DP only": 0, "Topology-aware chunk only": 1}
    panel_ref_rows.sort(
        key=lambda r: (
            _panel_order.get(r["label"], 3),
            int(m.group(1))
            if (m := re.search(r"K=(\d+)", r["label"])) and _fixed_k_label(r["label"])
            else 0,
            r["label"],
        )
    )
    plot_noise_sweep_three_lambdas(
        by_config,
        os.path.join(args.out_dir, "hybrid_noise_clip_sweep_lambda_panels.png"),
        auc_metric_name=auc_metric_title_suffix(args.auc_col),
        show_panel_titles=True,
        show_figure_title=False,
        reference_rows=panel_ref_rows,
    )
    print("Saved hybrid_noise_clip_sweep_lambda_panels.png (+ .pdf)")

    def _pad_lim(lo: float, hi: float, frac_pad: float = 0.04, abs_pad: float = 0.02):
        span = max(hi - lo, 1e-6)
        p = max(span * frac_pad, abs_pad)
        return lo - p, hi + p

    def _scatter_hybrid(ax, markersize: int, legend_handles: list | None) -> None:
        for i, row in by_plot.iterrows():
            label = rf"$\sigma$={row['dp_noise']}, $C$={row['dp_max_grad_norm']}"
            color = hybrid_colors[i]
            sc = ax.scatter(
                row["mean_accuracy"],
                row["mean_auc"],
                s=markersize,
                c=[color],
                edgecolors="black",
                linewidths=0.45,
                zorder=4,
            )
            if legend_handles is not None:
                legend_handles.append((sc, label))

    def _ref_in_zoom_window(
        acc: float, auc: float, x_lo: float, x_hi: float, y_lo: float, y_hi: float
    ) -> bool:
        return (x_lo <= acc <= x_hi) and (y_lo <= auc <= y_hi)

    # Width fixed; height is finalized after margins+legends so panel height matches the old 5.6" layout.
    _scatter_fig_w, _scatter_fig_h0 = 11.2, 5.6
    fig, (ax_full, ax_zoom) = plt.subplots(
        1,
        2,
        figsize=(_scatter_fig_w, _scatter_fig_h0),
        gridspec_kw={"width_ratios": [1.22, 1.0], "wspace": 0.28},
    )

    hybrid_legend_entries: list[tuple] = []
    _scatter_hybrid(ax_full, markersize=72, legend_handles=hybrid_legend_entries)

    ref_legend_handles: list = []
    ref_legend_labels: list[str] = []
    for ref in ref_points:
        sc_r = ax_full.scatter(
            ref["acc"],
            ref["auc"],
            s=200,
            marker="D",
            facecolors=ref["color"],
            edgecolors="black",
            linewidths=0.75,
            zorder=6,
            label=ref["label"],
        )
        ref_legend_handles.append(sc_r)
        ref_legend_labels.append(ref["label"])

    all_acc = by_plot["mean_accuracy"].tolist() + [r["acc"] for r in ref_points]
    all_auc = by_plot["mean_auc"].tolist() + [r["auc"] for r in ref_points]
    ax_full.set_xlim(*_pad_lim(min(all_acc), max(all_acc)))
    ax_full.set_ylim(*_pad_lim(min(all_auc), max(all_auc)))
    ax_full.set_xlabel(mean_accuracy_axis_label())
    ax_full.set_ylabel(mean_mia_auc_axis_label(args.auc_col))
    ax_full.set_title("Full context")
    ax_full.grid(True, alpha=0.3)

    _scatter_hybrid(ax_zoom, markersize=100, legend_handles=None)
    hx0, hx1 = float(by_plot["mean_accuracy"].min()), float(by_plot["mean_accuracy"].max())
    hy0, hy1 = float(by_plot["mean_auc"].min()), float(by_plot["mean_auc"].max())
    sx = max(hx1 - hx0, 1e-6)
    sy = max(hy1 - hy0, 1e-6)
    m = float(np.clip(args.scatter_inset_frac, 0.02, 0.45))
    x_zoom_lo, x_zoom_hi = hx0 - m * sx, hx1 + m * sx
    y_zoom_lo, y_zoom_hi = hy0 - m * sy, hy1 + m * sy
    ax_zoom.set_xlim(x_zoom_lo, x_zoom_hi)
    ax_zoom.set_ylim(y_zoom_lo, y_zoom_hi)

    # Fixed-K chunking reference diamonds (--extra-scatter-ref) in zoom when inside the zoom window
    any_std_in_zoom = False
    for ref in ref_points:
        if not ref.get("is_standard"):
            continue
        if not _ref_in_zoom_window(ref["acc"], ref["auc"], x_zoom_lo, x_zoom_hi, y_zoom_lo, y_zoom_hi):
            continue
        any_std_in_zoom = True
        ax_zoom.scatter(
            ref["acc"],
            ref["auc"],
            s=200,
            marker="D",
            facecolors=ref["color"],
            edgecolors="black",
            linewidths=0.75,
            zorder=6,
        )

    ax_zoom.set_xlabel(mean_accuracy_axis_label())
    ax_zoom.set_ylabel(mean_mia_auc_axis_label(args.auc_col))
    ax_zoom.set_title("Zoom: ChunkDP sweep")
    ax_zoom.grid(True, alpha=0.35)
    ax_zoom.tick_params(labelsize=9)

    # Scatter legend layout (tuned for synced/plots/noise_sweep_er008_with_standard_k_refs scatter).
    # Reference runs on top, ChunkDP below; stacked figure legends below the axes.
    h_hyb = [h for h, _ in hybrid_legend_entries]
    lab_hyb = [lb for _, lb in hybrid_legend_entries]
    _leg_kw_base = dict(
        loc="upper center",
        framealpha=0.96,
        bbox_transform=fig.transFigure,
    )
    _leg_colspacing_ref = 2.55
    _leg_borderpad_ref = 1.48
    _leg_handtextpad_ref = 0.75
    _leg_kw_ref = {
        **_leg_kw_base,
        "labelspacing": 2.02,
        "columnspacing": _leg_colspacing_ref,
        "handletextpad": _leg_handtextpad_ref,
        "borderpad": _leg_borderpad_ref,
    }
    # Legends live only in y ∈ [0, bottom_margin]; axes start at y = bottom_margin.
    # Do not use tight_layout here: the zoom panel makes tight_layout unreliable and legends
    # can end up on top of the axes. Stack from the physical bottom (small y).
    ncol_ref = min(4, max(2, len(ref_legend_handles))) if ref_legend_handles else 1
    n_ref_rows = (
        max(1, (len(ref_legend_handles) + ncol_ref - 1) // ncol_ref) if ref_legend_handles else 0
    )
    # ChunkDP: match Reference-runs legend width (ncol_ref × colspacing); more columns → fewer rows.
    if h_hyb and ref_legend_handles and n_ref_rows > 0:
        ncol_hyb = int(np.ceil(len(h_hyb) / n_ref_rows))
        ncol_hyb = int(np.clip(ncol_hyb, ncol_ref, 8))
    elif h_hyb:
        ncol_hyb = 4
    else:
        ncol_hyb = 1
    colspacing_hyb = _leg_colspacing_ref * ncol_ref / ncol_hyb
    _leg_kw_hyb = {
        **_leg_kw_base,
        "labelspacing": 0.42,
        "columnspacing": colspacing_hyb,
        "handletextpad": _leg_handtextpad_ref,
        "borderpad": _leg_borderpad_ref,
    }
    n_hyb_rows = max(1, (len(h_hyb) + ncol_hyb - 1) // ncol_hyb)
    H_h = min(0.17, 0.042 + 0.024 * n_hyb_rows)
    # Reference legend uses larger labelspacing between rows; budget extra height when n_ref_rows > 1.
    H_r = (
        (0.062 + 0.040 * n_ref_rows + 0.062 * max(0, n_ref_rows - 1) + 0.038)
        if ref_legend_handles
        else 0.0
    )
    gap_frac = -0.095
    pad_bot = 0.006
    # `upper center`: anchor y is the top of the legend; box extends downward by ~H_*.
    y_hybrid_top = pad_bot + H_h
    y_ref_top = y_hybrid_top + gap_frac + H_r if ref_legend_handles else y_hybrid_top
    # Leave space *above* the legend stack for x-axis tick labels + xlabel (they draw below the axes box).
    label_band = -0.06
    bottom_margin = min(0.72, max(0.34, y_ref_top + label_band))

    fig.legend(
        h_hyb,
        lab_hyb,
        bbox_to_anchor=(0.5, y_hybrid_top),
        ncol=ncol_hyb,
        fontsize=6.6,
        title=rf"ChunkDP ($\sigma$, $C$)",
        **_leg_kw_hyb,
    )
    if ref_legend_handles:
        fig.legend(
            ref_legend_handles,
            ref_legend_labels,
            bbox_to_anchor=(0.5, y_ref_top),
            ncol=ncol_ref,
            fontsize=7.0,
            title="Reference runs",
            **_leg_kw_ref,
        )

    fig.set_layout_engine(None)
    _scatter_top = 0.88
    fig.subplots_adjust(
        left=0.07,
        right=0.995,
        bottom=bottom_margin,
        top=_scatter_top,
        wspace=0.28,
    )
    # Tall bottom margin shrinks axes in *figure fraction*; grow figure height so plot area
    # matches the pre-legend ~5.6" effective panel (~(top−bottom) of the old tight_layout scatter).
    _ref_axes_y_frac = 0.74
    axes_y_frac = max(1e-3, _scatter_top - bottom_margin)
    target_axes_h_in = _scatter_fig_h0 * _ref_axes_y_frac
    scatter_fig_h = float(np.clip(target_axes_h_in / axes_y_frac, 6.0, 11.5))
    fig.set_size_inches(_scatter_fig_w, scatter_fig_h, forward=True)

    scatter_path = os.path.join(args.out_dir, "hybrid_noise_clip_sweep_scatter.png")
    _save_fig(fig, scatter_path)
    plt.close()
    print(f"Saved scatter: {scatter_path} (+ .pdf)")

    # Bar chart: accuracy and AUC by config
    if not args.no_bars:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        x = np.arange(len(by_config))
        w = 0.35
        labels = [f"σ={r['dp_noise']}\nC={r['dp_max_grad_norm']}" for _, r in by_config.iterrows()]
        acc_vals = by_config["mean_accuracy"].values
        acc_err = by_config["std_accuracy"].fillna(0).values
        ax1.bar(x - w / 2, acc_vals, w, yerr=acc_err, capsize=3, color="tab:blue")
        ax1.set_ylabel(mean_accuracy_axis_label())
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=15, ha="right")
        ax1.set_ylim(0, 1.05)
        auc_vals = by_config["mean_auc"].values
        auc_err = by_config["std_auc"].fillna(0).values
        ax2.bar(x + w / 2, auc_vals, w, yerr=auc_err, capsize=3, color="tab:red")
        ax2.set_ylabel(mean_mia_auc_axis_label(args.auc_col))
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=15, ha="right")
        ax2.set_ylim(0.4, 1.0)
        # no figure title (requested for thesis figures)
        plt.tight_layout()
        bar_path = os.path.join(args.out_dir, "hybrid_noise_clip_sweep_bars.png")
        _save_fig(fig2, bar_path)
        plt.close()
        print(f"Saved bars: {bar_path} (+ .pdf)")

    # Score bar chart: configs ordered by score (higher = better)
    if not args.no_score_plot:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        x = np.arange(len(by_config))
        labels = [f"σ={r['dp_noise']}, C={r['dp_max_grad_norm']}" for _, r in by_config.iterrows()]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(by_config))[::-1])  # best = darker
        bars = ax3.bar(x, by_config["score"].values, color=colors)
        ax3.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax3.set_ylabel(f"Score ((1−λ)u − λr); λ = {lam}")
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, rotation=25, ha="right")
        # no figure title (requested for thesis figures)
        plt.tight_layout()
        score_path = os.path.join(args.out_dir, "hybrid_noise_clip_sweep_score.png")
        _save_fig(fig3, score_path)
        plt.close()
        print(f"Saved score plot: {score_path} (+ .pdf)")

    print("Done.")


if __name__ == "__main__":
    main()
