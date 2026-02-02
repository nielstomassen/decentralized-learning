#!/usr/bin/env python3
"""
Analyze and plot correlation between node centrality measures and MIA AUC
from CSVs like:

  mnist_cnn_er_0.1053_alpha0.3_beta0.2_messagedelta_seed123.csv

Expected CSV columns (at least):
  round,node_id,degree,betweenness,closeness,k_core,...,auc

We do NOT average over nodes: each (file, node_id) is a sample.
We pick metrics at a fixed round (default 30) if present; otherwise the last
round where AUC is available.

Outputs:
  - aggregated_samples.csv
  - per-er_p correlation tables
  - scatter plots centrality vs auc per er_p and pooled

Usage:
  python analyze_centrality_auc.py \
    --results-glob "../../results/*.csv" \
    --round 30 \
    --out-dir "../../plots/centrality_auc"
"""

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


ER_REGEX = re.compile(r"(?:^|_)er_([0-9.]+)(?:_|$)", re.IGNORECASE)
SEED_REGEX = re.compile(r"seed(\d+)", re.IGNORECASE)


CENTRALITY_COLS = ["degree", "betweenness", "closeness", "k_core"]
METRIC_COLS = ["auc", "test_acc", "train_acc", "chunk_frac"]  # optional, if present


def parse_er_p(path: str) -> Optional[float]:
    m = ER_REGEX.search(os.path.basename(path))
    if not m:
        return None
    return float(m.group(1))


def parse_seed(path: str) -> Optional[int]:
    m = SEED_REGEX.search(os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def pick_round_rows(df: pd.DataFrame, target_round: int) -> Tuple[pd.DataFrame, int]:
    """
    Return one row per node_id at target_round if available (and AUC notna),
    else use last available round with AUC per node_id.

    Returns: (picked_df, used_round)
    used_round is the target_round if used; else -1 indicating mixed/last-per-node.
    """
    if "round" not in df.columns or "node_id" not in df.columns:
        raise KeyError("CSV must contain 'round' and 'node_id'")

    # Only keep rows with AUC present
    if "auc" not in df.columns:
        raise KeyError("CSV must contain 'auc'")

    df2 = df.copy()
    df2 = df2.dropna(subset=["auc"])
    if df2.empty:
        return df2, target_round

    # Try target round first
    df_target = df2[df2["round"] == target_round].copy()
    if not df_target.empty:
        # Keep one row per node_id (should already be one)
        df_target = df_target.sort_values(["node_id"]).drop_duplicates(subset=["node_id"], keep="last")
        return df_target, target_round

    # Otherwise pick last available (per node)
    # Sort by round then take last per node_id
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

    # If x or y is constant, correlation is undefined
    if np.allclose(x2, x2[0]) or np.allclose(y2, y2[0]):
        return out

    pr, pp = pearsonr(x2, y2)
    sr, sp = spearmanr(x2, y2)
    out.update({"pearson_r": float(pr), "pearson_p": float(pp), "spearman_r": float(sr), "spearman_p": float(sp)})
    return out


def scatter_plot(x, y, xlabel, ylabel, title, out_path, annotate: Optional[Dict[str, float]] = None):
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if annotate and annotate.get("n", 0) >= 3:
        t2 = (
            f"{title}\n"
            f"n={annotate['n']} | pearson r={annotate['pearson_r']:.3f} (p={annotate['pearson_p']:.3g}) | "
            f"spearman r={annotate['spearman_r']:.3f} (p={annotate['spearman_p']:.3g})"
        )
        plt.title(t2)
    else:
        plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def binned_mean_plot(x, y, xlabel, ylabel, title, out_path, n_bins: int = 6):
    """
    Bin x into quantiles, plot mean y per bin.
    Helpful when x has many ties / discrete values (degree, k_core).
    """
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty:
        return
    # if x has too few unique values, just skip
    if df["x"].nunique() < 3:
        return

    # quantile bins
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-glob", type=str, required=True, help="Glob for CSV files")
    ap.add_argument("--round", type=int, default=30, help="Round to take metrics from (else last available)")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for plots + csv")
    args = ap.parse_args()

    files = sorted(glob.glob(args.results_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.results_glob}")

    os.makedirs(args.out_dir, exist_ok=True)

    all_samples: List[pd.DataFrame] = []

    for f in files:
        raw = pd.read_csv(f)
        raw.columns = [c.strip() for c in raw.columns]

        # numeric conversion
        raw = ensure_numeric(raw, ["round", "node_id"] + CENTRALITY_COLS + METRIC_COLS)

        picked, used_round = pick_round_rows(raw, args.round)
        if picked.empty:
            print(f"[WARN] No AUC rows in {os.path.basename(f)}; skipping")
            continue

        picked["source_file"] = os.path.basename(f)
        picked["seed"] = parse_seed(f)
        picked["er_p"] = parse_er_p(f)
        picked["round_used"] = used_round if used_round != -1 else picked["round"].values  # keep actual

        # Keep only relevant columns if present
        keep = ["source_file", "seed", "er_p", "round", "node_id", "round_used"] + CENTRALITY_COLS
        for c in METRIC_COLS:
            if c in picked.columns:
                keep.append(c)
        picked = picked[keep].copy()

        all_samples.append(picked)

    if not all_samples:
        raise SystemExit("No usable samples found (no files had AUC rows).")

    samples = pd.concat(all_samples, ignore_index=True)

    # Save aggregated samples
    agg_path = os.path.join(args.out_dir, "aggregated_samples.csv")
    samples.to_csv(agg_path, index=False)
    print(f"[OK] Wrote {len(samples)} samples to {agg_path}")

    # Group by er_p (if er_p couldn't be parsed, it will be NaN)
    if samples["er_p"].notna().any():
        groups = list(samples.groupby("er_p"))
    else:
        groups = [(np.nan, samples)]

    # Correlation tables
    corr_rows = []
    for er_p, sdf in groups:
        for cent in CENTRALITY_COLS:
            if cent not in sdf.columns or "auc" not in sdf.columns:
                continue
            st = corr_stats(sdf[cent].to_numpy(dtype=float), sdf["auc"].to_numpy(dtype=float))
            corr_rows.append(
                {
                    "er_p": er_p,
                    "centrality": cent,
                    "n_samples": st["n"],
                    "pearson_r": st["pearson_r"],
                    "pearson_p": st["pearson_p"],
                    "spearman_r": st["spearman_r"],
                    "spearman_p": st["spearman_p"],
                }
            )

    corr_df = pd.DataFrame(corr_rows).sort_values(["er_p", "centrality"])
    corr_out = os.path.join(args.out_dir, "correlations_by_er_p.csv")
    corr_df.to_csv(corr_out, index=False)
    print(f"[OK] Wrote correlation table to {corr_out}")

    # Plot per er_p and pooled
    def do_plots(tag: str, sdf: pd.DataFrame):
        for cent in CENTRALITY_COLS:
            if cent not in sdf.columns or "auc" not in sdf.columns:
                continue
            x = sdf[cent].to_numpy(dtype=float)
            y = sdf["auc"].to_numpy(dtype=float)
            st = corr_stats(x, y)

            scatter_plot(
                x=x,
                y=y,
                xlabel=cent,
                ylabel="AUC",
                title=f"AUC vs {cent} ({tag})",
                out_path=os.path.join(args.out_dir, f"scatter_auc_vs_{cent}_{tag}.png"),
                annotate=st,
            )

            # Optional binned mean plot (useful for degree/k_core)
            binned_mean_plot(
                x=x,
                y=y,
                xlabel=cent,
                ylabel="AUC",
                title=f"AUC vs {cent} (binned, {tag})",
                out_path=os.path.join(args.out_dir, f"binned_auc_vs_{cent}_{tag}.png"),
            )

    # Per er_p plots
    if len(groups) > 1 and samples["er_p"].notna().any():
        for er_p, sdf in groups:
            tag = f"er_{er_p}"
            do_plots(tag, sdf)

    # Pooled plots
    do_plots("pooled", samples)

    print(f"[DONE] Plots + tables saved under: {args.out_dir}")


if __name__ == "__main__":
    main()
