#!/usr/bin/env python3
"""
Analyze per-node privacy leakage (MIA AUC) vs centrality metrics across runs.

This version supports the NEW CSV schema with aggregated attacker metrics, e.g.:
  avg_auc, max_auc, argmax_attacker,
  avg_chunk_frac, max_chunk_frac, argmax_chunk_attacker,
  epsilon, global_test_acc

Usage example:
  python maxauc_vs_degree.py \
    --results-glob "../../results/*.csv" \
    --round 30 \
    --out-dir ../../plots/ \
    --auc-col max_auc \
    --plot-global-test-acc \
    --xlim 0 20 \
    --ylim 0.4 0.9 \
    --y2lim 0.7 0.9
"""

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


ER_REGEX = re.compile(r"(?:^|_)er_([0-9.]+)(?:_|$)", re.IGNORECASE)
SEED_REGEX = re.compile(r"seed(\d+)", re.IGNORECASE)

CENTRALITY_COLS = ["degree", "betweenness", "closeness", "k_core"]

# New schema metric columns we may want to keep/average if present
METRIC_COLS = [
    "avg_auc",
    "max_auc",
    "avg_chunk_frac",
    "max_chunk_frac",
    "global_test_acc",
    "epsilon",
    # keep these too if you output them
    "argmax_attacker",
    "argmax_chunk_attacker",
]


def parse_er_p(path: str) -> Optional[float]:
    m = ER_REGEX.search(os.path.basename(path))
    return float(m.group(1)) if m else None


def parse_seed(path: str) -> Optional[int]:
    m = SEED_REGEX.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def pick_round_rows(df: pd.DataFrame, target_round: int, auc_col: str) -> Tuple[pd.DataFrame, int]:
    """
    Return one row per node_id at target_round if available (and auc_col notna),
    else use last available round with auc_col per node_id.

    Returns: (picked_df, used_round)
    used_round is the target_round if used; else -1 indicating last-per-node.
    """
    if "round" not in df.columns or "node_id" not in df.columns:
        raise KeyError("CSV must contain 'round' and 'node_id'")
    if auc_col not in df.columns:
        raise KeyError(f"CSV must contain '{auc_col}'")

    df2 = df.dropna(subset=[auc_col]).copy()
    if df2.empty:
        return df2, target_round

    df_target = df2[df2["round"] == target_round].copy()
    if not df_target.empty:
        df_target = (
            df_target.sort_values(["node_id"])
            .drop_duplicates(subset=["node_id"], keep="last")
            .reset_index(drop=True)
        )
        return df_target, target_round

    df_last = (
        df2.sort_values(["node_id", "round"])
        .groupby("node_id", as_index=False)
        .tail(1)
        .sort_values("node_id")
        .reset_index(drop=True)
    )
    return df_last, -1


def corr_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x2, y2 = x[mask], y[mask]
    out: Dict[str, float] = {
        "n": int(x2.size),
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_r": np.nan,
        "spearman_p": np.nan,
    }
    if x2.size < 3:
        return out
    if np.allclose(x2, x2[0]) or np.allclose(y2, y2[0]):
        return out
    pr, pp = pearsonr(x2, y2)
    sr, sp = spearmanr(x2, y2)
    out.update(
        {
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_r": float(sr),
            "spearman_p": float(sp),
        }
    )
    return out


def scatter_plot(
    x,
    y,
    xlabel,
    ylabel,
    title,
    out_path,
    annotate: Optional[Dict[str, float]] = None,
    y2: Optional[np.ndarray] = None,
    y2_label: str = "",
    y2_color: str = "tab:orange",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    y2lim: Optional[Tuple[float, float]] = None,
):
    """
    Scatter y vs x (one point per node).
    If y2 is provided, plot mean(y2) per unique x on a second y-axis.

    Axis fixing:
      - xlim fixes x-axis range
      - ylim fixes primary y-axis range
      - y2lim fixes secondary y-axis range (if plotted)
    """
    fig, ax1 = plt.subplots()

    ax1.scatter(x, y, alpha=0.8)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    if xlim is not None:
        ax1.set_xlim(xlim)
    if ylim is not None:
        ax1.set_ylim(ylim)

    if y2 is not None:
        x_arr = np.asarray(x, dtype=float)
        y2_arr = np.asarray(y2, dtype=float)

        mask = np.isfinite(x_arr) & np.isfinite(y2_arr)
        if np.any(mask):
            df2 = pd.DataFrame({"x": x_arr[mask], "y2": y2_arr[mask]})
            g = df2.groupby("x", as_index=False)["y2"].mean().sort_values("x")
            xs = g["x"].to_numpy()
            ys2 = g["y2"].to_numpy()

            ax2 = ax1.twinx()
            ax2.plot(
                xs,
                ys2,
                marker="o",
                linestyle="-",
                color=y2_color,
                linewidth=2,
                markersize=4,
                label=y2_label or "global_test_acc (mean per x)",
            )
            ax2.set_ylabel(y2_label or "global_test_acc (mean per x)", color=y2_color)
            ax2.tick_params(axis="y", colors=y2_color)

            if y2lim is not None:
                ax2.set_ylim(y2lim)

    if annotate and annotate.get("n", 0) >= 3:
        t2 = (
            f"{title}\n"
            f"n={annotate['n']} | pearson r={annotate['pearson_r']:.3f} (p={annotate['pearson_p']:.3g}) | "
            f"spearman r={annotate['spearman_r']:.3f} (p={annotate['spearman_p']:.3g})"
        )
        ax1.set_title(t2)
    else:
        ax1.set_title(title)

    ax1.grid(True, linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def binned_mean_plot(x, y, xlabel, ylabel, title, out_path, n_bins: int = 6):
    """
    Bin x into quantiles, plot mean y per bin (with std error bars).
    Helpful when x has many ties / discrete values (degree, k_core).
    """
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty:
        return
    if df["x"].nunique() < 3:
        return

    df["bin"] = pd.qcut(df["x"], q=min(n_bins, df["x"].nunique()), duplicates="drop")
    g = df.groupby("bin", observed=True)["y"]
    y_mean = g.mean().values
    y_std = g.std(ddof=0).values
    x_mid = np.arange(len(y_mean))

    plt.figure()
    plt.errorbar(x_mid, y_mean, yerr=y_std, fmt="o-", capsize=4)
    plt.xticks(x_mid, [str(b) for b in g.mean().index], rotation=30, ha="right")
    plt.xlabel(f"{xlabel} (quantile bins)")
    plt.ylabel(f"{ylabel} (mean ± std)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def seed_average(samples: pd.DataFrame, keep_argmax_cols: bool = False) -> pd.DataFrame:
    """
    Average over seeds per (er_p, node_id).

    Note: argmax columns are IDs; averaging them is meaningless.
    By default we drop them from averaging; optionally keep them as the first non-null.
    """
    if "er_p" not in samples.columns:
        samples["er_p"] = np.nan

    # Only average numeric metrics that make sense
    avg_candidates = [c for c in CENTRALITY_COLS + METRIC_COLS if c in samples.columns]
    # Remove argmax cols from mean aggregation
    argmax_cols = [c for c in ["argmax_attacker", "argmax_chunk_attacker"] if c in avg_candidates]
    avg_cols = [c for c in avg_candidates if c not in argmax_cols]

    def _seed_list(s):
        ss = [int(x) for x in s.dropna().unique().tolist()]
        ss.sort()
        return ",".join(map(str, ss))

    grouped = samples.groupby(["er_p", "node_id"], dropna=False)

    out = grouped[avg_cols].mean(numeric_only=True).reset_index()
    out["n_seeds"] = grouped["seed"].apply(lambda s: int(s.dropna().nunique())).values
    out["seeds"] = grouped["seed"].apply(_seed_list).values

    if "round_used" in samples.columns:
        out["round_used_min"] = grouped["round_used"].min().values
        out["round_used_max"] = grouped["round_used"].max().values

    if keep_argmax_cols:
        for c in argmax_cols:
            out[c] = grouped[c].apply(lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan).values

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-glob", type=str, required=True, help="Glob for CSV files")
    ap.add_argument("--round", type=int, default=30, help="Round to take metrics from (else last available)")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for plots + csv")

    ap.add_argument(
        "--auc-col",
        type=str,
        default="max_auc",
        choices=["avg_auc", "max_auc"],
        help="Which AUC column to use for correlations/plots.",
    )

    ap.add_argument(
        "--plot-global-test-acc",
        action="store_true",
        help="If present (and column exists), overlay global_test_acc on a second y-axis in scatter plots.",
    )

    # NEW: fixed axis limits for cross-graph comparability
    ap.add_argument("--xlim", type=float, nargs=2, default=None, help="Fix x-axis limits: min max")
    ap.add_argument("--ylim", type=float, nargs=2, default=None, help="Fix primary y-axis limits: min max")
    ap.add_argument(
        "--y2lim",
        type=float,
        nargs=2,
        default=None,
        help="Fix secondary y-axis (global_test_acc) limits: min max (only used if plotted).",
    )

    args = ap.parse_args()

    files = sorted(glob.glob(args.results_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.results_glob}")

    os.makedirs(args.out_dir, exist_ok=True)

    all_samples: List[pd.DataFrame] = []
    auc_col = args.auc_col

    for f in files:
        raw = pd.read_csv(f)
        raw.columns = [c.strip() for c in raw.columns]

        raw = ensure_numeric(raw, ["round", "node_id"] + CENTRALITY_COLS + METRIC_COLS)

        picked, used_round = pick_round_rows(raw, args.round, auc_col)
        if picked.empty:
            print(f"[WARN] No {auc_col} rows in {os.path.basename(f)}; skipping")
            continue

        picked["source_file"] = os.path.basename(f)
        picked["seed"] = parse_seed(f)
        picked["er_p"] = parse_er_p(f)

        # If we used last-per-node rows, keep the actual round in round_used
        picked["round_used"] = used_round if used_round != -1 else picked["round"].values

        keep = ["source_file", "seed", "er_p", "round", "node_id", "round_used"] + CENTRALITY_COLS
        for c in METRIC_COLS:
            if c in picked.columns:
                keep.append(c)
        picked = picked[keep].copy()

        all_samples.append(picked)

    if not all_samples:
        raise SystemExit("No usable samples found (no files had the selected AUC column).")

    samples_raw = pd.concat(all_samples, ignore_index=True)

    raw_path = os.path.join(args.out_dir, "aggregated_samples_raw.csv")
    samples_raw.to_csv(raw_path, index=False)
    print(f"[OK] Wrote {len(samples_raw)} raw samples to {raw_path}")

    samples = seed_average(samples_raw, keep_argmax_cols=False)

    avg_path = os.path.join(args.out_dir, "aggregated_samples_seedavg.csv")
    samples.to_csv(avg_path, index=False)
    print(f"[OK] Wrote {len(samples)} seed-averaged samples to {avg_path}")

    # Group by ER p if we parsed it
    if "er_p" in samples.columns and samples["er_p"].notna().any():
        groups = list(samples.groupby("er_p"))
    else:
        groups = [(np.nan, samples)]

    corr_rows = []
    for er_p, sdf in groups:
        for cent in CENTRALITY_COLS:
            if cent not in sdf.columns or auc_col not in sdf.columns:
                continue
            st = corr_stats(sdf[cent].to_numpy(dtype=float), sdf[auc_col].to_numpy(dtype=float))
            corr_rows.append(
                {
                    "er_p": er_p,
                    "centrality": cent,
                    "auc_col": auc_col,
                    "n_samples": st["n"],
                    "pearson_r": st["pearson_r"],
                    "pearson_p": st["pearson_p"],
                    "spearman_r": st["spearman_r"],
                    "spearman_p": st["spearman_p"],
                }
            )

    corr_df = pd.DataFrame(corr_rows).sort_values(["er_p", "centrality"])
    corr_out = os.path.join(args.out_dir, f"correlations_by_er_p_{auc_col}.csv")
    corr_df.to_csv(corr_out, index=False)
    print(f"[OK] Wrote correlation table to {corr_out}")

    def do_plots(tag: str, sdf: pd.DataFrame):
        has_gta = args.plot_global_test_acc and ("global_test_acc" in sdf.columns)

        for cent in CENTRALITY_COLS:
            if cent not in sdf.columns or auc_col not in sdf.columns:
                continue

            x = sdf[cent].to_numpy(dtype=float)
            y = sdf[auc_col].to_numpy(dtype=float)
            st = corr_stats(x, y)

            y2 = None
            y2_label = ""
            if has_gta:
                y2 = sdf["global_test_acc"].to_numpy(dtype=float)
                y2_label = "global_test_acc (seed-avg)"

            scatter_plot(
                x=x,
                y=y,
                xlabel=cent,
                ylabel=f"{auc_col} (seed-avg per node)",
                title=f"{auc_col} vs {cent} ({tag})",
                out_path=os.path.join(args.out_dir, f"scatter_{auc_col}_vs_{cent}_{tag}.png"),
                annotate=st,
                y2=y2,
                y2_label=y2_label,
                xlim=tuple(args.xlim) if args.xlim is not None else None,
                ylim=tuple(args.ylim) if args.ylim is not None else None,
                y2lim=tuple(args.y2lim) if args.y2lim is not None else None,
            )

            binned_mean_plot(
                x=x,
                y=y,
                xlabel=cent,
                ylabel=f"{auc_col} (seed-avg per node)",
                title=f"{auc_col} vs {cent} (binned, {tag})",
                out_path=os.path.join(args.out_dir, f"binned_{auc_col}_vs_{cent}_{tag}.png"),
            )

    if len(groups) > 1 and "er_p" in samples.columns and samples["er_p"].notna().any():
        for er_p, sdf in groups:
            do_plots(f"er_{er_p}", sdf)

    do_plots("pooled", samples)

    print(f"[DONE] Plots + tables saved under: {args.out_dir}")


if __name__ == "__main__":
    main()