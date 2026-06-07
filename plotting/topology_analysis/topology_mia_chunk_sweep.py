#!/usr/bin/env python3
"""
Aggregate baseline vs chunk-only topology sweep CSVs and produce thesis figures.
Used for topology analysis results in the thesis
Reads synced/results/.../baseline_chunk_topology_sweep/<folder>/*.csv

Outputs (PDF + PNG):
  - topology_deterministic_bars.{pdf,png}
  - topology_regular_degree.{pdf,png}
  - topology_privacy_utility.{pdf,png}  (baseline vs chunking on one axes)
  - topology_er_by_p.{pdf,png}  (ER sweep vs. edge probability p, if ER folders exist)

Usage:
  python3 -m plotting.topology_analysis.topology_mia_chunk_sweep \
    --root results/topology_analysis \
    --out-dir plots/topology_analysis
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from statistics import mean, stdev

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

CHUNK_RE = re.compile(r"_chunk([01])")

YLABEL_TOP5_ACC = "Mean Top-5 Test Accuracy"
YLABEL_MIA_MAX = "Mean Maximum MIA AUC"


def _fnum(x: str) -> float:
    try:
        return float(x) if x not in ("", None) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load_last_round_means(path: str) -> dict | None:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    last = max(int(r["round"]) for r in rows)
    rdf = [r for r in rows if int(r["round"]) == last]
    accs = [_fnum(r["global_test_acc"]) for r in rdf]
    avg_aucs = [_fnum(r["avg_auc"]) for r in rdf]
    max_aucs = [_fnum(r["max_auc"]) for r in rdf]
    accs = [a for a in accs if a == a]
    avg_aucs = [a for a in avg_aucs if a == a]
    max_aucs = [a for a in max_aucs if a == a]
    m = CHUNK_RE.search(os.path.basename(path))
    chunk = int(m.group(1)) if m else 0
    seed_m = re.search(r"seed(\d+)", path)
    seed = int(seed_m.group(1)) if seed_m else -1
    return {
        "chunk": chunk,
        "seed": seed,
        "acc": mean(accs) if accs else float("nan"),
        "avg_auc": mean(avg_aucs) if avg_aucs else float("nan"),
        "max_auc": mean(max_aucs) if max_aucs else float("nan"),
    }


def collect(root: str) -> list[dict]:
    out = []
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.csv"))):
            m = load_last_round_means(p)
            if m is None:
                continue
            m["folder"] = sub
            m["path"] = p
            out.append(m)
    return out


def agg_mean_std(vals: list[float]) -> tuple[float, float]:
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan")
    mu = mean(vals)
    sig = stdev(vals) if len(vals) > 1 else 0.0
    return mu, sig


def thesis_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def save_both(fig, out_dir: str, stem: str):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        print(f"Wrote {p}")


def plot_deterministic(rows: list[dict], out_dir: str):
    folders = ["ring", "star", "grid", "full"]
    labels = ["Ring", "Star", "Grid", "Fully connected"]
    by_f_ch = defaultdict(list)
    for r in rows:
        by_f_ch[(r["folder"], r["chunk"])].append(r)

    x = np.arange(len(folders))
    w = 0.36
    acc_b, acc_e, auc_b, auc_e = [], [], [], []
    acc_c, acc_ce, auc_c, auc_ce = [], [], [], []
    for f in folders:
        for ch, acc_list, auc_list, e_acc, e_auc in (
            (0, acc_b, auc_b, acc_e, auc_e),
            (1, acc_c, auc_c, acc_ce, auc_ce),
        ):
            vs = by_f_ch.get((f, ch), [])
            am, asd = agg_mean_std([v["acc"] for v in vs])
            um, usd = agg_mean_std([v["max_auc"] for v in vs])
            acc_list.append(am)
            e_acc.append(asd)
            auc_list.append(um)
            e_auc.append(usd)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax1.bar(x - w / 2, acc_b, w, yerr=acc_e, label="Baseline", color="#2c5282", capsize=3, ecolor="#333")
    ax1.bar(x + w / 2, acc_c, w, yerr=acc_ce, label="Chunking", color="#c05621", capsize=3, ecolor="#333")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=12, ha="right")
    ax1.set_ylabel(YLABEL_TOP5_ACC)
    ax1.set_ylim(0.2, 0.55)
    ax1.legend(loc="upper right")
    ax1.set_title("(a) Utility")

    ax2.bar(x - w / 2, auc_b, w, yerr=auc_e, label="Baseline", color="#2c5282", capsize=3, ecolor="#333")
    ax2.bar(x + w / 2, auc_c, w, yerr=auc_ce, label="Chunking", color="#c05621", capsize=3, ecolor="#333")
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random MIA")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=12, ha="right")
    ax2.set_ylabel(YLABEL_MIA_MAX)
    ax2.set_ylim(0.45, 1.02)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_title("(b) Membership leakage")

    plt.tight_layout()
    save_both(fig, out_dir, "topology_deterministic_bars")
    plt.close()


def plot_regular(rows: list[dict], out_dir: str):
    by_f_ch = defaultdict(list)
    for r in rows:
        by_f_ch[(r["folder"], r["chunk"])].append(r)

    degrees = [3, 10, 25]
    folders = [f"regular_d{d}" for d in degrees]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))
    for ch, name, color, marker in (
        (0, "Baseline", "#2c5282", "o"),
        (1, "Chunking", "#c05621", "s"),
    ):
        acc_m, acc_s, auc_m, auc_s = [], [], [], []
        for f in folders:
            vs = by_f_ch.get((f, ch), [])
            am, asd = agg_mean_std([v["acc"] for v in vs])
            um, usd = agg_mean_std([v["max_auc"] for v in vs])
            acc_m.append(am)
            acc_s.append(asd)
            auc_m.append(um)
            auc_s.append(usd)
        ax1.errorbar(degrees, acc_m, yerr=acc_s, fmt=f"-{marker}", capsize=4, label=name, color=color, linewidth=2, markersize=7)
        ax2.errorbar(degrees, auc_m, yerr=auc_s, fmt=f"-{marker}", capsize=4, label=name, color=color, linewidth=2, markersize=7)

    ax1.set_xlabel(r"Degree $d$ in $d$-regular graph")
    ax1.set_ylabel(YLABEL_TOP5_ACC)
    ax1.set_xticks(degrees)
    ax1.set_ylim(0.42, 0.52)
    ax1.legend()
    ax1.set_title(r"(a) Utility vs. regular degree")

    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel(r"Degree $d$ in $d$-regular graph")
    ax2.set_ylabel(YLABEL_MIA_MAX)
    ax2.set_xticks(degrees)
    ax2.set_ylim(0.45, 1.02)
    ax2.legend()
    ax2.set_title(r"(b) Leakage vs. regular degree")

    plt.tight_layout()
    save_both(fig, out_dir, "topology_regular_degree")
    plt.close()


def _jitter_overlapping_pts(pts: list[tuple]) -> list[tuple]:
    """Tiny planar jitter for identical rounded (acc, auc); preserves any trailing fields on each tuple."""
    buckets: dict[tuple[float, float], list[int]] = defaultdict(list)
    for i, p in enumerate(pts):
        key = (round(p[0], 3), round(p[1], 3))
        buckets[key].append(i)
    out = list(pts)
    for idxs in buckets.values():
        if len(idxs) <= 1:
            continue
        for j, i in enumerate(idxs):
            t = out[i]
            am, um = t[0], t[1]
            rest = t[2:]
            out[i] = (am + j * 0.004, um + j * 0.006, *rest)
    return out


def _folder_sort_key(folder: str) -> tuple:
    if folder.startswith("er_p_"):
        return (4, float(folder.replace("er_p_", "")))
    if folder.startswith("regular_d"):
        return (3, int(folder.replace("regular_d", "")))
    order = {"ring": 0, "star": 1, "grid": 2, "full": 2.5}
    return (2, order.get(folder, 99), folder)


def plot_privacy_utility(rows: list[dict], out_dir: str):
    """Single axes: color = graph family, marker = baseline (circle) vs chunking (square)."""
    by_f_ch = defaultdict(list)
    for r in rows:
        by_f_ch[(r["folder"], r["chunk"])].append(r)

    def short_label(folder: str) -> str:
        if folder == "full":
            return "Full"
        if folder in ("ring", "star", "grid"):
            return folder.capitalize()
        if folder.startswith("regular_d"):
            return f"Reg-{folder.split('_d')[1]}"
        if folder.startswith("er_p_"):
            p = folder.replace("er_p_", "")
            return rf"ER $p={p}$"
        return folder

    all_folders = sorted({f for (f, _) in by_f_ch.keys()}, key=_folder_sort_key)
    n_f = max(len(all_folders), 1)
    tab = mpl.colormaps["tab20"].resampled(n_f)
    folder_color = {f: tab(i / max(n_f - 1, 1)) for i, f in enumerate(all_folders)}

    fig, ax = plt.subplots(figsize=(7.6, 6.35))
    # bottom margin: x-axis label above legend (figure legend sits below the label)
    fig.subplots_adjust(bottom=0.23, top=0.96, left=0.11, right=0.97)
    xmin, xmax = 0.22, 0.54
    ymin, ymax = 0.52, 1.01

    pts: list[tuple] = []
    for (folder, ckey), vs in sorted(by_f_ch.items()):
        if not vs:
            continue
        am, _ = agg_mean_std([v["acc"] for v in vs])
        um, _ = agg_mean_std([v["max_auc"] for v in vs])
        n = len(vs)
        pts.append((am, um, short_label(folder), n, folder, ckey))
    pts = _jitter_overlapping_pts(pts)

    present_union: set[str] = set()
    for am, um, lab, n, folder, ckey in pts:
        marker = "o" if ckey == 0 else "s"
        c = folder_color[folder]
        ax.scatter(
            am,
            um,
            s=58 + 9 * min(n, 5),
            c=[c],
            marker=marker,
            alpha=0.92,
            edgecolors="0.2",
            linewidths=0.65,
            zorder=3,
        )
        present_union.add(folder)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(YLABEL_TOP5_ACC)
    ax.set_ylabel(YLABEL_MIA_MAX)

    ncol_leg = 6 if n_f > 8 else 5
    fam_handles = [
        mlines.Line2D(
            [],
            [],
            color=folder_color[f],
            marker="o",
            linestyle="None",
            markersize=7.0,
            markeredgecolor="0.2",
            markeredgewidth=0.55,
            label=short_label(f),
        )
        for f in all_folders
        if f in present_union
    ]
    if fam_handles:
        fig.legend(
            handles=fam_handles,
            fontsize=6.7,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.14),
            ncol=ncol_leg,
            frameon=True,
            framealpha=0.97,
            edgecolor="0.75",
            columnspacing=0.85,
            handlelength=1.0,
            borderpad=0.25,
            labelspacing=0.28,
        )

    save_both(fig, out_dir, "topology_privacy_utility")
    plt.close()


def plot_er_by_p(rows: list[dict], out_dir: str):
    """Network-level accuracy and MIA AUC vs. ER edge probability p (all completed runs)."""
    er_folders = sorted(f for f in set(r["folder"] for r in rows) if f.startswith("er_p_"))
    if not er_folders:
        return
    by_f_ch = defaultdict(list)
    for r in rows:
        by_f_ch[(r["folder"], r["chunk"])].append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))
    p_vals = []
    for f in er_folders:
        try:
            p_vals.append(float(f.replace("er_p_", "")))
        except ValueError:
            continue
    order = sorted(zip(er_folders, p_vals), key=lambda t: t[1])
    er_folders = [t[0] for t in order]
    p_vals = [t[1] for t in order]
    x = np.arange(len(er_folders))
    w = 0.35

    for ax, key, ylab, ylim in (
        (ax1, "acc", YLABEL_TOP5_ACC, None),
        (ax2, "max_auc", YLABEL_MIA_MAX, (0.45, 1.02)),
    ):
        b_m, b_e, c_m, c_e = [], [], [], []
        for f in er_folders:
            vb = by_f_ch.get((f, 0), [])
            vc = by_f_ch.get((f, 1), [])
            am, sd = agg_mean_std([v[key] for v in vb])
            b_m.append(am)
            b_e.append(sd)
            am, sd = agg_mean_std([v[key] for v in vc])
            c_m.append(am)
            c_e.append(sd)
        ax.bar(x - w / 2, b_m, w, yerr=b_e, label="Baseline", color="#2c5282", capsize=3)
        ax.bar(x + w / 2, c_m, w, yerr=c_e, label="Chunking", color="#c05621", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in p_vals])
        ax.set_xlabel(r"ER edge probability $p$")
        ax.set_ylabel(ylab)
        if ylim:
            ax.set_ylim(*ylim)
        if key == "max_auc":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.legend(fontsize=8)

    ax1.set_title("(a) Utility")
    ax2.set_title("(b) Leakage")
    plt.tight_layout()
    save_both(fig, out_dir, "topology_er_by_p")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="synced/results/cifar100/baseline_chunk_topology_sweep",
    )
    ap.add_argument("--out-dir", type=str, default="Master_Thesis/figures")
    args = ap.parse_args()

    thesis_style()
    rows = collect(args.root)
    if not rows:
        raise SystemExit(f"No CSV rows under {args.root}")

    plot_deterministic(rows, args.out_dir)
    plot_regular(rows, args.out_dir)
    plot_privacy_utility(rows, args.out_dir)
    plot_er_by_p(rows, args.out_dir)

    # Summary print for thesis table
    by_f_ch = defaultdict(list)
    for r in rows:
        by_f_ch[(r["folder"], r["chunk"])].append(r)
    print("\n# Summary (mean ± std over seeds)")
    for key in sorted(by_f_ch.keys(), key=lambda k: (k[0], k[1])):
        vs = by_f_ch[key]
        am, ase = agg_mean_std([v["acc"] for v in vs])
        umx, usex = agg_mean_std([v["max_auc"] for v in vs])
        print(f"  {key[0]} chunk={key[1]} n={len(vs)} acc={am:.4f}±{ase:.4f} max_auc={umx:.4f}±{usex:.4f}")


if __name__ == "__main__":
    main()
