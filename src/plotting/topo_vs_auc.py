#!/usr/bin/env python3
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Accept:
#   alpha0.6
#   alpha_0.6
#   alpha-0.6
ALPHA_REGEX = re.compile(r"alpha[^0-9]*([0-9]*\.?[0-9]+)", re.IGNORECASE)

# Accept:
#   seed123
#   seed_123
#   seed-123
SEED_REGEX = re.compile(r"seed[^0-9]*([0-9]+)", re.IGNORECASE)


def parse_alpha_from_filename(path: str) -> float:
    base = os.path.basename(path)
    m = ALPHA_REGEX.search(base)
    if not m:
        raise ValueError(f"Could not find alpha in filename: {path}")
    return float(m.group(1))


def parse_seed_from_filename(path: str) -> int | None:
    base = os.path.basename(path)
    m = SEED_REGEX.search(base)
    if not m:
        return None
    return int(m.group(1))


TOPO_TOKEN_REGEX = re.compile(r"(^|[\/_\-\.])(full|ring)(?=([\/_\-\.]|$))", re.IGNORECASE)

def parse_topology_from_filename(path: str) -> str:
    base = os.path.basename(path)

    # 1) Prefer basename (least ambiguous)
    m = TOPO_TOKEN_REGEX.search(base)
    if m:
        return m.group(2).lower()

    # 2) Fallback: search the whole path (directory names)
    m = TOPO_TOKEN_REGEX.search(path)
    if m:
        return m.group(2).lower()

    raise ValueError(f"Could not find topology (ring/full) in filename or path: {path}")


def summarize_by_round_mean_over_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input CSV: per-node per-round rows with columns like:
      round,node_id,train_acc,test_acc,auc

    Output: one row per round with mean train_acc/test_acc/auc over nodes.
    """
    required = {"round", "node_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Coerce numeric (empty strings -> NaN)
    for c in ["round", "node_id", "train_acc", "test_acc", "auc"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    agg_cols = [c for c in ["train_acc", "test_acc", "auc"] if c in df.columns]
    out = (
        df.groupby("round", as_index=False)[agg_cols]
        .mean()
        .sort_values("round")
        .reset_index(drop=True)
    )
    return out


def pick_fixed_or_last_row(
    df_round: pd.DataFrame,
    fixed_round: int,
    metric_cols=("auc", "test_acc"),
) -> pd.Series:
    cols_present = [c for c in metric_cols if c in df_round.columns]
    if not cols_present:
        raise KeyError(f"None of the metric cols {metric_cols} present in df.")

    df2 = df_round.dropna(subset=cols_present, how="all").sort_values("round")
    if df2.empty:
        raise ValueError("No usable metric rows after dropping NaNs.")

    if fixed_round in df2["round"].values:
        return df2.loc[df2["round"] == fixed_round].iloc[0]
    return df2.iloc[-1]


def load_runs(files: list[str], fixed_round: int) -> pd.DataFrame:
    rows = []
    for f in files:
        alpha = parse_alpha_from_filename(f)
        topo = parse_topology_from_filename(f)
        seed = parse_seed_from_filename(f)

        raw = pd.read_csv(f)
        per_round = summarize_by_round_mean_over_nodes(raw)
        picked = pick_fixed_or_last_row(per_round, fixed_round, metric_cols=("auc", "test_acc"))

        rows.append(
            {
                "alpha": alpha,
                "topology": topo,
                "seed": seed,
                "picked_round": int(picked["round"]),
                "auc": float(picked["auc"]) if "auc" in picked and pd.notna(picked["auc"]) else np.nan,
                "test_acc": float(picked["test_acc"]) if "test_acc" in picked and pd.notna(picked["test_acc"]) else np.nan,
                "source_file": os.path.basename(f),
            }
        )
    return pd.DataFrame(rows)


def aggregate_over_seeds(runs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across runs (seeds) for each (topology, alpha).
    Output columns:
      topology, alpha, n_runs_auc, mean_auc, std_auc, n_runs_acc, mean_acc, std_acc
    """

    def _agg(g: pd.DataFrame) -> pd.Series:
        auc_vals = g["auc"].dropna().to_numpy(dtype=float)
        acc_vals = g["test_acc"].dropna().to_numpy(dtype=float)
        return pd.Series(
            {
                "n_runs_auc": int(auc_vals.size),
                "mean_auc": float(np.mean(auc_vals)) if auc_vals.size else np.nan,
                "std_auc": float(np.std(auc_vals, ddof=0)) if auc_vals.size else np.nan,
                "n_runs_acc": int(acc_vals.size),
                "mean_acc": float(np.mean(acc_vals)) if acc_vals.size else np.nan,
                "std_acc": float(np.std(acc_vals, ddof=0)) if acc_vals.size else np.nan,
            }
        )

    out = (
        runs_df.groupby(["topology", "alpha"], as_index=False)
        .apply(_agg, include_groups=False)
        .reset_index(drop=True)
        .sort_values(["topology", "alpha"])
        .reset_index(drop=True)
    )
    return out


def plot_auc_vs_alpha_by_topology(
    agg: pd.DataFrame,
    out_dir: str,
    fixed_round: int,
    topologies: list[str],
):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()

    for topo in topologies:
        df = agg[agg["topology"] == topo].dropna(subset=["alpha", "mean_auc", "std_auc"]).sort_values("alpha")
        if df.empty:
            print(f"[WARN] No plottable rows for topology={topo}. Skipping.")
            continue

        plt.errorbar(
            df["alpha"].values,
            df["mean_auc"].values,
            yerr=df["std_auc"].values,
            fmt="o-",
            capsize=4,
            label=topo,
        )

    plt.xlabel("alpha")
    plt.ylabel("Mean AUC (mean over nodes per run, then mean over seeds)")
    plt.title(f"AUC vs alpha @ round {fixed_round} (by topology)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    out_path = os.path.join(out_dir, "auc_vs_alpha_by_topology.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot AUC vs alpha for selected topologies (e.g., ring vs full), "
            "averaging over nodes within each run and over seeds across runs.\n\n"
            "Expected filename tokens: alpha_0.6 (or alpha0.6), topology_full/ring, seed_123.\n"
            "Expected CSV columns: round,node_id,auc (test_acc optional)."
        )
    )
    parser.add_argument(
        "--results-glob",
        type=str,
        default="../../results/*.csv",
        help="Glob pattern to find CSV result files.",
    )
    parser.add_argument(
        "--fixed-round",
        type=int,
        default=30,
        help="Round index to use; if not present, uses last usable round per file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../plots/auc_by_alpha_topology",
        help="Directory to save aggregated CSVs and plots.",
    )
    parser.add_argument(
        "--topologies",
        type=str,
        default="ring,full",
        help="Comma-separated list of topologies to include (e.g. 'ring,full').",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.results_glob))
    if not files:
        raise SystemExit(f"No files matched pattern: {args.results_glob}")

    topologies = [t.strip().lower() for t in args.topologies.split(",") if t.strip()]
    print(f"Found {len(files)} result files.")
    print(f"Including topologies: {topologies}")

    runs_df = load_runs(files, fixed_round=args.fixed_round)
    os.makedirs(args.output_dir, exist_ok=True)
    runs_df.to_csv(os.path.join(args.output_dir, "runs_picked_metrics.csv"), index=False)

    agg = aggregate_over_seeds(runs_df)
    agg.to_csv(os.path.join(args.output_dir, "auc_acc_by_topology_alpha.csv"), index=False)

    plot_auc_vs_alpha_by_topology(agg, args.output_dir, args.fixed_round, topologies)

    print(
        f"\nSaved:"
        f"\n  - {os.path.join(args.output_dir, 'runs_picked_metrics.csv')}"
        f"\n  - {os.path.join(args.output_dir, 'auc_acc_by_topology_alpha.csv')}"
        f"\n  - {os.path.join(args.output_dir, 'auc_vs_alpha_by_topology.png')}"
        f"\n\nOutput dir: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
