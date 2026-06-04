#!/usr/bin/env python3
"""
Deployment score S_λ = (1−λ)u − λ r with r = max(0, 2a−1) for multiple λ values.

Plots used in the thesis: three panels tied to utility-oriented / balanced / privacy-first
narratives (Section methodology scalar score).
Used in hybrid ablation section in the thesis. 
Called by hybrid_privacy_tradeoff.py so no need to run this script separately.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chunkdp_labels import (
    CHUNKDP_CONDITION,
    chunkdp_xtick_label,
    deployment_score_xlabel,
    deployment_score_ylabel,
)


# Align with Master_Thesis/sections/methodology.tex narrative anchors
DEFAULT_LAMBDAS = (0.25, 0.50, 0.75)


def _lam_suffix(lam: float) -> str:
    """Stable column suffix, e.g. 0.25 -> '0_25'."""
    return f"{lam:.2f}".replace(".", "_")


PANEL_TITLES = (
    r"$\lambda=0.25$: utility-oriented",
    r"$\lambda=0.50$: balanced",
    r"$\lambda=0.75$: privacy-first",
)
PANEL_SUBTITLES = (
    "(e.g. consumer / on-device)",
    "(e.g. automotive / fleet)",
    "(e.g. clinical / legal)",
)


def privacy_risk_from_auc(mean_auc: np.ndarray | pd.Series) -> np.ndarray:
    return np.maximum(0.0, 2.0 * np.asarray(mean_auc, dtype=float) - 1.0)


def deployment_score(
    mean_accuracy: np.ndarray | pd.Series,
    mean_auc: np.ndarray | pd.Series,
    lam: float,
) -> np.ndarray:
    lam = float(np.clip(lam, 0.0, 1.0))
    u = np.asarray(mean_accuracy, dtype=float)
    r = privacy_risk_from_auc(mean_auc)
    return (1.0 - lam) * u - lam * r


def _save_fig(fig, path_png: str, dpi: int = 150) -> None:
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    base, _ = os.path.splitext(path_png)
    fig.savefig(base + ".pdf", bbox_inches="tight")


def export_sweep_lambda_table(
    df: pd.DataFrame,
    out_csv: str,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
) -> pd.DataFrame:
    """Add S_λ columns and per-λ ranks; save CSV."""
    out = df.copy()
    pr = privacy_risk_from_auc(out["mean_auc"])
    out["privacy_risk_r"] = pr
    for lam in lambdas:
        suf = _lam_suffix(lam)
        out[f"S_lambda_{suf}"] = deployment_score(out["mean_accuracy"], out["mean_auc"], lam)
    for lam in lambdas:
        suf = _lam_suffix(lam)
        scol = f"S_lambda_{suf}"
        out[f"rank_lambda_{suf}"] = out[scol].rank(ascending=False, method="min").astype(int)
    out.to_csv(out_csv, index=False)
    return out


def plot_noise_sweep_three_lambdas(
    df: pd.DataFrame,
    out_path_png: str,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    auc_metric_name: str = "max MIA AUC",
    show_panel_titles: bool = True,
    show_figure_title: bool = True,
    reference_rows: list[dict] | None = None,
) -> None:
    """
    One row × three columns: horizontal bars of S_λ per (σ,C), sorted best-first in each panel.
    df columns: mean_accuracy, mean_auc, dp_noise, dp_max_grad_norm
    """
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.2), sharey=False)
    cmap = plt.colormaps.get_cmap("viridis")
    df = df.reset_index(drop=True)

    for ax, lam, title, sub in zip(axes, lambdas, PANEL_TITLES, PANEL_SUBTITLES):
        scores = deployment_score(df["mean_accuracy"], df["mean_auc"], lam)
        entries: list[dict] = []
        for i, row in df.iterrows():
            entries.append(
                {
                    "score": float(scores[i]),
                    "label": rf"$\sigma$={row['dp_noise']}, $C$={row['dp_max_grad_norm']}",
                    "is_ref": False,
                    "ref_color": None,
                }
            )
        if reference_rows:
            for ref in reference_rows:
                lbl = str(ref["label"])
                if lbl == "Baseline":
                    c_ref = "#2ca02c"
                elif lbl == "DP only":
                    c_ref = "#9467bd"
                elif lbl.startswith("Topology-aware"):
                    c_ref = "#ff7f0e"
                else:
                    # Fixed-K chunking (and legacy "Standard chunking" labels) plus any other refs
                    c_ref = "#66c2a5"
                entries.append(
                    {
                        "score": float(deployment_score(ref["mean_accuracy"], ref["mean_auc"], lam)),
                        "label": lbl,
                        "is_ref": True,
                        "ref_color": c_ref,
                    }
                )

        entries = sorted(entries, key=lambda e: float(e["score"]), reverse=True)
        y_pos = np.arange(len(entries))
        s_sorted = np.asarray([e["score"] for e in entries], dtype=float)
        lbl_sorted = [str(e["label"]) for e in entries]
        grad = cmap(np.linspace(0.25, 0.9, len(entries))[::-1])
        colors = [e["ref_color"] if e["is_ref"] else grad[i] for i, e in enumerate(entries)]

        ax.barh(y_pos, s_sorted, color=colors, edgecolor="black", linewidth=0.35)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(lbl_sorted, fontsize=8)
        for tick, e in zip(ax.get_yticklabels(), entries):
            if e["is_ref"]:
                tick.set_fontweight("bold")
        ax.set_xlabel(deployment_score_xlabel(), fontsize=9)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        if show_panel_titles:
            ax.set_title(f"{title}\n{sub}", fontsize=10)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)

    if show_figure_title:
        fig.suptitle(
            f"ChunkDP noise × clip sweep: ranking $(\\sigma, C)$ by deployment score ({auc_metric_name})",
            fontsize=11,
            y=1.02,
        )
    plt.tight_layout()
    _save_fig(fig, out_path_png)
    plt.close()


def pick_sweep_argmax_per_lambda(
    df: pd.DataFrame,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
) -> tuple[list[pd.Series], list[int]]:
    """
    For each λ, the (σ,C) row that maximizes S_λ. Returns (winner_row_per_λ, argmax_index_per_λ).
    """
    df = df.reset_index(drop=True)
    u = df["mean_accuracy"].to_numpy(dtype=float)
    a = df["mean_auc"].to_numpy(dtype=float)
    winners: list[pd.Series] = []
    indices: list[int] = []
    for lam in lambdas:
        s = deployment_score(u, a, lam)
        idx = int(np.argmax(s))
        indices.append(idx)
        winners.append(df.iloc[idx])
    return winners, indices


def export_sweep_lambda_optima_meta(
    df: pd.DataFrame,
    out_csv: str,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
) -> pd.DataFrame:
    """One row per λ: which (σ,C) maximizes S_λ and scores under all three λ."""
    winners, idxs = pick_sweep_argmax_per_lambda(df, lambdas)
    rows = []
    for k, lam in enumerate(lambdas):
        row = winners[k]
        entry = {
            "optim_for_lambda": lam,
            "argmax_row_index": idxs[k],
            "dp_noise": row["dp_noise"],
            "dp_max_grad_norm": row["dp_max_grad_norm"],
            "mean_accuracy": row["mean_accuracy"],
            "mean_auc": row["mean_auc"],
        }
        for lam2 in lambdas:
            suf = _lam_suffix(lam2)
            entry[f"S_lambda_{suf}"] = float(
                deployment_score(row["mean_accuracy"], row["mean_auc"], lam2)
            )
        rows.append(entry)
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


def plot_sweep_optima_grouped_three_lambdas(
    df: pd.DataFrame,
    out_path_png: str,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    auc_metric_name: str = "max MIA AUC",
    show_title: bool = True,
) -> None:
    """
    Same layout as the ablation λ-grouped figure: x = three groups, each group is the
    sweep cell that maximizes S_λ for that column's λ; three bars show S_{λ'} evaluated
    at that cell for λ' ∈ {0.25, 0.5, 0.75}.
    """
    winners, _ = pick_sweep_argmax_per_lambda(df, lambdas)
    n = len(lambdas)
    x = np.arange(n, dtype=float)
    w = 0.24
    fig, ax = plt.subplots(figsize=(9, 4.6))

    for j, lam_eval in enumerate(lambdas):
        heights = [
            float(
                deployment_score(
                    winners[i]["mean_accuracy"],
                    winners[i]["mean_auc"],
                    lam_eval,
                )
            )
            for i in range(n)
        ]
        offset = (j - 1) * w
        ax.bar(
            x + offset,
            heights,
            width=w,
            label=rf"$\lambda={lam_eval:g}$",
            edgecolor="black",
            linewidth=0.4,
        )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    xlabs = [
        rf"max $S_{{{lam:g}}}$" + "\n" + rf"$\sigma$={row['dp_noise']}, $C$={row['dp_max_grad_norm']}"
        for lam, row in zip(lambdas, winners)
    ]
    ax.set_xticklabels(xlabs, fontsize=8.5)
    ax.set_ylabel(deployment_score_ylabel())
    ax.legend(title=r"$\lambda$ in $S_\lambda$", fontsize=8.5, title_fontsize=8.5)
    if show_title:
        ax.set_title(
            f"ChunkDP $(\\sigma, C)$ sweep: deployment scores at $\\lambda$-specific argmax configs\n"
            f"({auc_metric_name})"
        )
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, out_path_png)
    plt.close()


def plot_ablation_grouped_three_lambdas(
    df: pd.DataFrame,
    out_path_png: str,
    condition_order: list[str] | None = None,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    auc_metric_name: str = "max MIA AUC",
    show_title: bool = True,
) -> None:
    """
    Grouped vertical bars: x = ablation condition, 3 bars per condition for S_λ.
    df: columns condition, mean_accuracy, mean_auc
    """
    if condition_order is None:
        condition_order = [
            "No DP, no chunk (baseline)",
            "DP only",
            "Chunk only",
            CHUNKDP_CONDITION,
        ]
    df = df.set_index("condition").reindex(condition_order).reset_index()
    df = df.dropna(subset=["mean_accuracy", "mean_auc"])

    x = np.arange(len(df))
    w = 0.24
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    display_labels = [chunkdp_xtick_label(c) for c in df["condition"].astype(str).tolist()]
    for i, lam in enumerate(lambdas):
        scores = deployment_score(df["mean_accuracy"], df["mean_auc"], lam)
        offset = (i - 1) * w
        ax.bar(
            x + offset,
            scores,
            width=w,
            label=rf"$\lambda={lam:g}$",
            edgecolor="black",
            linewidth=0.4,
        )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(deployment_score_ylabel())
    ax.legend(title="Tradeoff weight", fontsize=9, title_fontsize=9)
    if show_title:
        ax.set_title(f"ChunkDP ablation: deployment scores ({auc_metric_name})")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, out_path_png)
    plt.close()


def export_ablation_lambda_table(
    df: pd.DataFrame,
    out_csv: str,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
) -> pd.DataFrame:
    out = df[["condition", "mean_accuracy", "mean_auc"]].copy()
    out["condition"] = out["condition"].astype(str)
    for lam in lambdas:
        suf = _lam_suffix(lam)
        out[f"S_lambda_{suf}"] = deployment_score(out["mean_accuracy"], out["mean_auc"], lam)
    out.to_csv(out_csv, index=False)
    return out
