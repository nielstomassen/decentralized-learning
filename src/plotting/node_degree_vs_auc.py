#!/usr/bin/env python3
"""
node_degree_auc_noavg.py

Node-instance (NO averaging over nodes, NO averaging over seeds) analysis.

Each input CSV is assumed to have (at least) these columns:
  round,node_id,degree,auc
(Other columns may exist.)

For each file (run/seed):
  - extract ONE AUC per node (either at a fixed round with fallback-to-last, or best AUC over rounds)
Then:
  - correlate degree with AUC using *all node instances across all runs* (so n ~= n_files * n_nodes),
    optionally separately per group (e.g., per er_p).

This avoids the common pitfall for ER graphs where node_id is not comparable across seeds.

Outputs (in output-dir):
  all_node_instances.csv    # one row per node per file
  correlations.csv          # Pearson/Spearman per group
  scatter_degree_vs_auc__*.png

Example:
  python node_degree_auc_noavg.py \
    --results-glob "../../results/*.csv" \
    --pick fixed --fixed-round 30 \
    --group-by er_p \
    --output-dir "../../plots/degree_auc_noavg"
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- filename parsing -----------------
# Match patterns like "..._er_0.1053_..." or "..._er-0.1053_..."
ER_REGEX = re.compile(r"(?:^|_)er[_-]?([0-9]*\.?[0-9]+)(?:_|$)", re.IGNORECASE)
ALPHA_REGEX = re.compile(r"(?:^|_)alpha([0-9]*\.?[0-9]+)(?:_|$)", re.IGNORECASE)
BETA_REGEX = re.compile(r"(?:^|_)beta([0-9]*\.?[0-9]+)(?:_|$)", re.IGNORECASE)
SEED_REGEX = re.compile(r"(?:^|_)seed(\d+)(?:_|$)", re.IGNORECASE)


def _parse_float(regex: re.Pattern, path: str) -> float | None:
    m = regex.search(os.path.basename(path))
    return float(m.group(1)) if m else None


def parse_er_p_from_filename(path: str) -> float | None:
    return _parse_float(ER_REGEX, path)


def parse_alpha_from_filename(path: str) -> float | None:
    return _parse_float(ALPHA_REGEX, path)


def parse_beta_from_filename(path: str) -> float | None:
    return _parse_float(BETA_REGEX, path)


def parse_seed_from_filename(path: str) -> int | None:
    m = SEED_REGEX.search(os.path.basename(path))
    return int(m.group(1)) if m else None


# ----------------- stats helpers -----------------
def pearsonr_np(x: np.ndarray, y: np.ndarray):
    """Return (r, p) if scipy is available, else (r, nan)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.nan, np.nan
    try:
        from scipy.stats import pearsonr  # type: ignore

        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        r = np.corrcoef(x, y)[0, 1]
        return float(r), np.nan


def spearmanr_np(x: np.ndarray, y: np.ndarray):
    """Return (rho, p) if scipy is available, else (rho, nan) via rank+pearson."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.nan, np.nan
    try:
        from scipy.stats import spearmanr  # type: ignore

        rho, p = spearmanr(x, y)
        return float(rho), float(p)
    except Exception:
        rx = pd.Series(x).rank(method="average").to_numpy()
        ry = pd.Series(y).rank(method="average").to_numpy()
        rho = np.corrcoef(rx, ry)[0, 1]
        return float(rho), np.nan


# ----------------- core extraction -----------------
def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def extract_node_auc_fixed_round(raw: pd.DataFrame, fixed_round: int) -> pd.DataFrame:
    """
    For each node_id in THIS FILE:
      - choose row at fixed_round if present
      - else choose last available (highest round) with auc present
    Returns one row per node (for THIS file):
      node_id, degree, picked_round, auc
    """
    required = {"round", "node_id", "degree", "auc"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = raw.copy()
    df = _ensure_numeric(df, ["round", "node_id", "degree", "auc"])

    # keep only rows where auc exists
    df = df.dropna(subset=["node_id", "round", "auc"])

    rows = []
    for node_id, g in df.groupby("node_id"):
        g = g.sort_values("round")
        if g.empty:
            continue

        if fixed_round in g["round"].values:
            row = g.loc[g["round"] == fixed_round].iloc[0]
        else:
            row = g.iloc[-1]

        rows.append(
            {
                "node_id": int(node_id),
                "degree": float(row["degree"]),
                "picked_round": int(row["round"]),
                "auc": float(row["auc"]),
            }
        )

    return pd.DataFrame(rows)


def extract_node_auc_best(raw: pd.DataFrame) -> pd.DataFrame:
    """
    For each node_id in THIS FILE:
      - pick the round where auc is maximal (per node)
    Returns one row per node (for THIS file):
      node_id, degree, picked_round, auc
    """
    required = {"round", "node_id", "degree", "auc"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = raw.copy()
    df = _ensure_numeric(df, ["round", "node_id", "degree", "auc"])
    df = df.dropna(subset=["node_id", "round", "auc"])

    idx = df.groupby("node_id")["auc"].idxmax()
    out = df.loc[idx, ["node_id", "degree", "round", "auc"]].copy()
    out.rename(columns={"round": "picked_round"}, inplace=True)

    out["node_id"] = out["node_id"].astype(int)
    out["picked_round"] = out["picked_round"].astype(int)
    out["degree"] = out["degree"].astype(float)
    out["auc"] = out["auc"].astype(float)

    return out.reset_index(drop=True)


def load_all_node_instances(files: list[str], pick: str, fixed_round: int) -> pd.DataFrame:
    """
    Returns one row per node per file (node INSTANCE):
      node_id, degree, auc, picked_round, alpha, beta, er_p, seed, source_file
    """
    all_rows = []
    for f in files:
        alpha = parse_alpha_from_filename(f)
        beta = parse_beta_from_filename(f)
        er_p = parse_er_p_from_filename(f)
        seed = parse_seed_from_filename(f)

        raw = pd.read_csv(f)

        if pick == "best":
            node_df = extract_node_auc_best(raw)
        else:
            node_df = extract_node_auc_fixed_round(raw, fixed_round)

        node_df["alpha"] = alpha
        node_df["beta"] = beta
        node_df["er_p"] = er_p
        node_df["seed"] = seed
        node_df["source_file"] = os.path.basename(f)

        all_rows.append(node_df)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


# ----------------- correlation + plotting -----------------
def correlations_by_group(node_instances: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Correlate degree vs auc using ALL node instances.
    If group_cols non-empty, do it per group.
    """
    if not group_cols:
        groups = [((), node_instances)]
    else:
        groups = list(node_instances.groupby(group_cols, dropna=False))

    rows = []
    for key, g in groups:
        dfp = g.dropna(subset=["degree", "auc"])
        x = dfp["degree"].to_numpy(dtype=float)
        y = dfp["auc"].to_numpy(dtype=float)

        r, p = pearsonr_np(x, y)
        rho, sp = spearmanr_np(x, y)

        row = {
            "n_samples": int(len(dfp)),  # THIS is the important n now (≈ n_files * n_nodes)
            "pearson_r": r,
            "pearson_p": p,
            "spearman_r": rho,
            "spearman_p": sp,
        }

        if group_cols:
            if not isinstance(key, tuple):
                key = (key,)
            for c, v in zip(group_cols, key):
                row[c] = v

        rows.append(row)

    # keep group cols first
    cols = list(group_cols) + ["n_samples", "pearson_r", "pearson_p", "spearman_r", "spearman_p"]
    out = pd.DataFrame(rows)
    return out[cols] if set(cols).issubset(out.columns) else out


def _group_label(group_cols: list[str], key) -> str:
    if not group_cols:
        return "global"
    if not isinstance(key, tuple):
        key = (key,)
    parts = []
    for c, v in zip(group_cols, key):
        parts.append(f"{c}={v}")
    return "__".join(parts).replace("/", "_")


def plot_scatter_degree_vs_auc(node_instances: pd.DataFrame, group_cols: list[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    if not group_cols:
        groups = [((), node_instances)]
    else:
        groups = list(node_instances.groupby(group_cols, dropna=False))

    for key, g in groups:
        label = _group_label(group_cols, key)
        out_path = os.path.join(out_dir, f"scatter_degree_vs_auc__{label}.png")

        dfp = g.dropna(subset=["degree", "auc"])
        if dfp.empty:
            print(f"[WARN] No data to plot for group={label}. Skipping.")
            continue

        r, _ = pearsonr_np(dfp["degree"].to_numpy(), dfp["auc"].to_numpy())
        rho, _ = spearmanr_np(dfp["degree"].to_numpy(), dfp["auc"].to_numpy())

        plt.figure()
        plt.scatter(dfp["degree"], dfp["auc"], alpha=0.6)
        plt.xlabel("Node degree")
        plt.ylabel("Node AUC (one per node per run)")
        plt.title(f"Degree vs AUC ({label})\nPearson r={r:.3f}, Spearman ρ={rho:.3f}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()


# ----------------- main -----------------
def main():
    parser = argparse.ArgumentParser(
        description="Correlate node degree with AUC using ALL node instances (no averaging across seeds)."
    )
    parser.add_argument(
        "--results-glob",
        type=str,
        default="../../results/*.csv",
        help="Glob pattern to find CSV result files.",
    )
    parser.add_argument(
        "--pick",
        type=str,
        choices=["fixed", "best"],
        default="fixed",
        help="How to pick per-node AUC from per-round rows: fixed round (fallback to last) or best over rounds.",
    )
    parser.add_argument(
        "--fixed-round",
        type=int,
        default=30,
        help="Fixed round used when --pick=fixed.",
    )
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=[],
        help="Group correlations/plots by any of: alpha beta er_p (space-separated). Empty = global.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../plots/degree_auc_noavg",
        help="Directory to save CSVs and plots.",
    )
    args = parser.parse_args()

    valid_group_cols = {"alpha", "beta", "er_p"}
    group_cols = [c for c in args.group_by if c in valid_group_cols]

    files = sorted(glob.glob(args.results_glob))
    if not files:
        raise SystemExit(f"No files matched pattern: {args.results_glob}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Found {len(files)} files")

    node_instances = load_all_node_instances(files, pick=args.pick, fixed_round=args.fixed_round)
    if node_instances.empty:
        raise SystemExit(
            "No node instances extracted. Check your CSV columns (need round,node_id,degree,auc)."
        )

    # Save instances table
    inst_path = os.path.join(args.output_dir, "all_node_instances.csv")
    node_instances.to_csv(inst_path, index=False)

    # Correlations (using ALL node instances)
    corr = correlations_by_group(node_instances, group_cols=group_cols)
    corr_path = os.path.join(args.output_dir, "correlations.csv")
    corr.to_csv(corr_path, index=False)

    # Scatter plots
    plot_scatter_degree_vs_auc(node_instances, group_cols=group_cols, out_dir=args.output_dir)

    print(f"\nSaved to: {args.output_dir}")
    print(f" - all_node_instances.csv (one row per node per run): {inst_path}")
    print(f" - correlations.csv (degree↔AUC, all node instances): {corr_path}")
    print(" - scatter_degree_vs_auc__*.png")


if __name__ == "__main__":
    main()
