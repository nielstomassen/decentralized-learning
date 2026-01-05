#!/usr/bin/env python
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ALPHA_REGEX = re.compile(r"alpha([0-9.]+)", re.IGNORECASE)
SEED_REGEX = re.compile(r"seed(\d+)", re.IGNORECASE)


def parse_alpha_from_filename(path: str) -> float:
    m = ALPHA_REGEX.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Could not find 'alphaX' in filename: {path}")
    return float(m.group(1))


def parse_seed_from_filename(path: str) -> int | None:
    m = SEED_REGEX.search(os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))


def summarize_by_round_mean_over_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input CSV: columns include round,node_id,train_acc,test_acc,auc (per node per round)
    Output: one row per round with mean train_acc/test_acc/auc over nodes (NaNs ignored).
    """
    required = {"round", "node_id", "test_acc", "auc"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Ensure numeric (your CSV has empty strings early on -> becomes NaN)
    for c in ["round", "node_id", "train_acc", "test_acc", "auc"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Mean over nodes per round (ignores NaNs by default)
    agg_cols = [c for c in ["train_acc", "test_acc", "auc"] if c in df.columns]
    round_means = (
        df.groupby("round", as_index=False)[agg_cols]
        .mean()
        .sort_values("round")
        .reset_index(drop=True)
    )

    return round_means


def load_results(files: list[str]) -> dict[float, list[pd.DataFrame]]:
    """
    Group CSVs by alpha.
    Each CSV is converted to per-round mean-over-nodes first.
    Returns: dict alpha -> list of DataFrames (one per seed/run), each with columns round,test_acc,auc,...
    """
    grouped: dict[float, list[pd.DataFrame]] = defaultdict(list)
    for f in files:
        alpha = parse_alpha_from_filename(f)
        raw = pd.read_csv(f)
        df = summarize_by_round_mean_over_nodes(raw)
        df["source_file"] = os.path.basename(f)
        grouped[alpha].append(df)
    return grouped


def _pick_fixed_or_last_row(df: pd.DataFrame, fixed_round: int) -> pd.Series:
    # Drop rounds where metrics are all NaN (common in early rounds in your example)
    metric_cols = [c for c in ["test_acc", "auc"] if c in df.columns]
    df2 = df.dropna(subset=metric_cols, how="all").sort_values("round")
    if df2.empty:
        raise ValueError(f"No usable metric rows found in {df['source_file'].iloc[0]}")

    if fixed_round in df2["round"].values:
        return df2.loc[df2["round"] == fixed_round].iloc[0]
    return df2.iloc[-1]


def aggregate_final_round(grouped: dict[float, list[pd.DataFrame]], fixed_round: int):
    """
    For each alpha and run:
      - compute per-round mean over nodes already
      - take metrics at fixed_round if present else last available
    Then mean/std across runs.
    """
    rows_acc = []
    rows_auc = []

    for alpha, runs in sorted(grouped.items(), key=lambda x: x[0]):
        acc_vals = []
        auc_vals = []

        for df in runs:
            row = _pick_fixed_or_last_row(df, fixed_round)

            if "test_acc" not in row.index or "auc" not in row.index:
                raise KeyError(
                    f"Expected 'test_acc' and 'auc' after aggregation in {df['source_file'].iloc[0]}"
                )

            if pd.notna(row["test_acc"]):
                acc_vals.append(float(row["test_acc"]))
            if pd.notna(row["auc"]):
                auc_vals.append(float(row["auc"]))

        acc_vals = np.array(acc_vals, dtype=float)
        auc_vals = np.array(auc_vals, dtype=float)

        rows_acc.append(
            {
                "alpha": alpha,
                "mean_acc": float(acc_vals.mean()) if acc_vals.size else np.nan,
                "std_acc": float(acc_vals.std(ddof=0)) if acc_vals.size else np.nan,
                "n_runs": int(acc_vals.size),
            }
        )
        rows_auc.append(
            {
                "alpha": alpha,
                "mean_auc": float(auc_vals.mean()) if auc_vals.size else np.nan,
                "std_auc": float(auc_vals.std(ddof=0)) if auc_vals.size else np.nan,
                "n_runs": int(auc_vals.size),
            }
        )

    return pd.DataFrame(rows_acc), pd.DataFrame(rows_auc)


def aggregate_best_round(grouped: dict[float, list[pd.DataFrame]]):
    """
    For each alpha and run:
      - find the round that maximizes *mean test_acc (over nodes)*
      - record that best_acc and the corresponding (mean-over-nodes) auc at that round
    Then mean/std across runs.
    """
    rows_acc = []
    rows_auc = []

    for alpha, runs in sorted(grouped.items(), key=lambda x: x[0]):
        best_acc_vals = []
        auc_at_best_vals = []

        for df in runs:
            metric_df = df.dropna(subset=["test_acc", "auc"], how="any").sort_values("round")
            if metric_df.empty:
                raise ValueError(f"No rows with both test_acc and auc in {df['source_file'].iloc[0]}")

            idx = metric_df["test_acc"].idxmax()
            row = metric_df.loc[idx]
            best_acc_vals.append(float(row["test_acc"]))
            auc_at_best_vals.append(float(row["auc"]))

        best_acc_vals = np.array(best_acc_vals, dtype=float)
        auc_at_best_vals = np.array(auc_at_best_vals, dtype=float)

        rows_acc.append(
            {
                "alpha": alpha,
                "mean_best_acc": float(best_acc_vals.mean()),
                "std_best_acc": float(best_acc_vals.std(ddof=0)),
                "n_runs": int(best_acc_vals.size),
            }
        )
        rows_auc.append(
            {
                "alpha": alpha,
                "mean_auc_at_best": float(auc_at_best_vals.mean()),
                "std_auc_at_best": float(auc_at_best_vals.std(ddof=0)),
                "n_runs": int(auc_at_best_vals.size),
            }
        )

    return pd.DataFrame(rows_acc), pd.DataFrame(rows_auc)


def plot_with_errorbars(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    y_err_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: str,
):
    dfp = df.dropna(subset=[x_col, y_col, y_err_col])
    if dfp.empty:
        print(f"[WARN] Nothing to plot for {title} (all NaN). Skipping: {out_path}")
        return

    plt.figure()
    plt.errorbar(
        dfp[x_col].values,
        dfp[y_col].values,
        yerr=dfp[y_err_col].values,
        fmt="o-",
        capsize=4,
    )
    # plt.xscale("log")  # optional
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate experiment runs over different alpha values and "
            "plot test accuracy and MIA AUC as a function of alpha.\n"
            "Supports per-node-per-round CSVs: round,node_id,train_acc,test_acc,auc"
        )
    )
    parser.add_argument(
        "--results-glob",
        type=str,
        default="../../results/*alpha*.csv",
        help=(
            "Glob pattern to find CSV result files. "
            "Filenames must contain 'alphaX' (e.g. alpha0.1). "
            "Each CSV should have columns: round,node_id,train_acc,test_acc,auc."
        ),
    )
    parser.add_argument(
        "--fixed-round",
        type=int,
        default=30,
        help="Round index to use for 'final' metrics (if larger than max round, uses last usable round).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../plots/alpha_sweep",
        help="Directory to save aggregated plots.",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.results_glob))
    if not files:
        raise SystemExit(f"No files matched pattern: {args.results_glob}")

    print(f"Found {len(files)} result files:")
    for f in files:
        seed = parse_seed_from_filename(f)
        print("  -", f, f"(seed={seed})" if seed is not None else "")

    grouped = load_results(files)

    df_acc_final, df_auc_final = aggregate_final_round(grouped, args.fixed_round)
    df_best_acc, df_auc_at_best = aggregate_best_round(grouped)

    os.makedirs(args.output_dir, exist_ok=True)

    df_acc_final.to_csv(os.path.join(args.output_dir, "final_round_accuracy.csv"), index=False)
    df_auc_final.to_csv(os.path.join(args.output_dir, "final_round_auc.csv"), index=False)
    df_best_acc.to_csv(os.path.join(args.output_dir, "best_round_accuracy.csv"), index=False)
    df_auc_at_best.to_csv(os.path.join(args.output_dir, "auc_at_best_round.csv"), index=False)

    plot_with_errorbars(
        df_auc_final,
        x_col="alpha",
        y_col="mean_auc",
        y_err_col="std_auc",
        xlabel=r"Dirichlet $\alpha$",
        ylabel="Mean node AUC (per run, then mean over seeds)",
        title=f"MIA AUC vs alpha (mean over nodes @ round {args.fixed_round})",
        out_path=os.path.join(args.output_dir, f"auc_vs_alpha_round{args.fixed_round}.png"),
    )

    # (Optional) keep the acc plots too
    plot_with_errorbars(
        df_acc_final,
        x_col="alpha",
        y_col="mean_acc",
        y_err_col="std_acc",
        xlabel=r"Dirichlet $\alpha$",
        ylabel="Mean node test accuracy",
        title=f"Test accuracy vs alpha (mean over nodes @ round {args.fixed_round})",
        out_path=os.path.join(args.output_dir, f"acc_vs_alpha_round{args.fixed_round}.png"),
    )

    plot_with_errorbars(
        df_auc_at_best,
        x_col="alpha",
        y_col="mean_auc_at_best",
        y_err_col="std_auc_at_best",
        xlabel=r"Dirichlet $\alpha$",
        ylabel="Mean node AUC",
        title="MIA AUC vs alpha (at best mean test accuracy round)",
        out_path=os.path.join(args.output_dir, "auc_vs_alpha_at_best_round.png"),
    )

    plot_with_errorbars(
        df_best_acc,
        x_col="alpha",
        y_col="mean_best_acc",
        y_err_col="std_best_acc",
        xlabel=r"Dirichlet $\alpha$",
        ylabel="Mean node test accuracy",
        title="Best mean test accuracy vs alpha",
        out_path=os.path.join(args.output_dir, "acc_vs_alpha_best_round.png"),
    )

    print(f"\nSaved aggregated CSVs and plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
