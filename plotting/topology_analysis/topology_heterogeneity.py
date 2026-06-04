#!/usr/bin/env python3
"""
Node-level heterogeneity analysis for baseline vs chunk topology sweep.
Used for topology analysis results in the thesis
- Star: hub (max degree) vs leaves (degree 1).
- Grid: degree classes 2 / 3 / 4 (corner / edge / interior on the 10×10 grid).
- ER: fixed degree bins per $p$ (see ER_DEGREE_BIN_SPEC in source); one faceted figure row per $p$.
- Ring & full: spread of per-node MIA metrics (no degree strata).

Outputs:
  plots/topology_star_hub_leaf.{pdf,png}
  plots/topology_grid_degree_roles.{pdf,png}
  plots/topology_er_degree_bins.{pdf,png}
  plots/topology_star_ring_auc_violin.{pdf,png}
  tables/*.tex  (booktabs tables for \\input)

Usage:
  python3 -m plotting/topology_analysis/topology_heterogeneity \\
    --root results/topology_analysis \\
    --out-dir-figures plots \\
    --out-dir-tables tables
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from statistics import mean, pstdev, stdev

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

CHUNK_RE = re.compile(r"_chunk([01])")

YLABEL_TOP5_ACC = "Mean Top-5 Test Accuracy"
YLABEL_MIA_MAX = "Mean Maximum MIA AUC"
YLABEL_MIA_MAX_PER_NODE = "Per-node Maximum MIA AUC"

# Single-metric leakage figures use worst-case (max) node-level MIA AUC; tables keep avg + max columns.
TARGET_ER_BINS = 7  # fallback when an ER folder is not in ER_DEGREE_BIN_SPEC
MIN_NODES_IN_BIN_PER_RUN = 3

# Fixed integer degree bins per ER sweep folder. Keys must match result subfolder names (e.g. er_p_0.04).
# half_open: degrees with lo <= d < hi_excl; hi_excl None => tail d >= lo (same as ge lo).
# Bins partition the supported range: low / mid / ... / high tail, with row labels = actual degree spans.
ER_DEGREE_BIN_SPEC: dict[str, list[dict]] = {
    "er_p_0.04": [
        {"kind": "eq", "d": 1, "label": "1"},
        {"kind": "eq", "d": 2, "label": "2"},
        {"kind": "eq", "d": 3, "label": "3"},
        {"kind": "ge", "lo": 4, "label": r"$\geq 4$"},
    ],
    # [2,6), [6,9), [9,11), [11,∞) — four bands from sparse-ER low degree through a high tail.
    "er_p_0.08": [
        {"kind": "half_open", "lo": 2, "hi_excl": 6, "label": "2--5"},
        {"kind": "half_open", "lo": 6, "hi_excl": 9, "label": "6--8"},
        {"kind": "half_open", "lo": 9, "hi_excl": 11, "label": "9--10"},
        {"kind": "half_open", "lo": 11, "hi_excl": None, "label": r"$\geq 11$"},
    ],
    # [7,12), [12,16), [16,20), [20,∞)
    "er_p_0.16": [
        {"kind": "half_open", "lo": 7, "hi_excl": 12, "label": "7--11"},
        {"kind": "half_open", "lo": 12, "hi_excl": 16, "label": "12--15"},
        {"kind": "half_open", "lo": 16, "hi_excl": 20, "label": "16--19"},
        {"kind": "half_open", "lo": 20, "hi_excl": None, "label": r"$\geq 20$"},
    ],
    # [20,27), [27,32), [32,37), [37,∞)
    "er_p_0.32": [
        {"kind": "half_open", "lo": 20, "hi_excl": 27, "label": "20--26"},
        {"kind": "half_open", "lo": 27, "hi_excl": 32, "label": "27--31"},
        {"kind": "half_open", "lo": 32, "hi_excl": 37, "label": "32--36"},
        {"kind": "half_open", "lo": 37, "hi_excl": None, "label": r"$\geq 37$"},
    ],
}


def _fixed_er_bin_match(deg: int, entry: dict) -> bool:
    k = entry["kind"]
    if k == "eq":
        return deg == entry["d"]
    if k == "ge":
        return deg >= entry["lo"]
    if k == "half_open":
        lo, hi_excl = entry["lo"], entry["hi_excl"]
        if hi_excl is None:
            return deg >= lo
        return lo <= deg < hi_excl
    raise ValueError(f"unknown bin entry kind: {k!r}")


def _fnum(x: str) -> float:
    try:
        return float(x) if x not in ("", None) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load_last_round_rows(path: str) -> tuple[list[dict], int]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], 0
    last = max(int(r["round"]) for r in rows)
    rdf = [r for r in rows if int(r["round"]) == last]
    return rdf, last


def parse_meta(path: str) -> dict:
    base = os.path.basename(path)
    m = CHUNK_RE.search(base)
    chunk = int(m.group(1)) if m else 0
    seed_m = re.search(r"seed(\d+)", base)
    seed = int(seed_m.group(1)) if seed_m else -1
    return {"chunk": chunk, "seed": seed, "path": path}


def mean_metric(rows: list[dict], col: str) -> float:
    vals = [_fnum(r[col]) for r in rows]
    vals = [v for v in vals if v == v]
    return mean(vals) if vals else float("nan")


def ms(vals: list[float]) -> tuple[float, float]:
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan")
    mu = mean(vals)
    if len(vals) == 1:
        return mu, 0.0
    return mu, stdev(vals)


def thesis_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
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


def collect_by_folder(root: str) -> dict[str, list[tuple[dict, list[dict]]]]:
    """folder -> list of (meta, last_round_rows)."""
    out: dict[str, list[tuple[dict, list[dict]]]] = defaultdict(list)
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.csv"))):
            rows, _ = load_last_round_rows(p)
            if not rows:
                continue
            meta = parse_meta(p)
            meta["folder"] = sub
            out[sub].append((meta, rows))
    return out


def star_hub_leaf(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    max_deg = max(int(r["degree"]) for r in rows)
    hub = [r for r in rows if int(r["degree"]) == max_deg]
    leaves = [r for r in rows if int(r["degree"]) == 1]
    return hub, leaves


def pooled_degrees(data: list[tuple[dict, list[dict]]]) -> np.ndarray:
    out = []
    for _meta, rows in data:
        for r in rows:
            out.append(int(r["degree"]))
    return np.array(out, dtype=int)


def equal_frequency_degree_bin_ranges(pooled: np.ndarray, k: int = TARGET_ER_BINS) -> list[tuple[int, int]]:
    """~k disjoint degree intervals (inclusive) that split pooled nodes into near-equal counts.

    Uses cumulative counts over ordered unique degrees (histogram), not raw quantiles of degree
    values---otherwise dense ER graphs collapse to one wide bin.
    """
    pooled = np.asarray(pooled, dtype=int)
    if pooled.size == 0:
        return []
    dmin, dmax = int(pooled.min()), int(pooled.max())
    if dmin == dmax:
        return [(dmin, dmax)]
    bc = np.bincount(pooled - dmin, minlength=dmax - dmin + 1)
    u = np.flatnonzero(bc) + dmin
    c = bc[bc > 0]
    cum = np.cumsum(c)
    ntot = float(cum[-1])
    k_eff = min(max(3, k), 7)
    split_idx = [0]
    for i in range(1, k_eff):
        target = ntot * i / k_eff
        j = int(np.searchsorted(cum, max(1.0, target), side="left"))
        j = min(max(0, j), len(u) - 1)
        split_idx.append(j)
    split_idx.append(len(u) - 1)
    split_idx = sorted(set(split_idx))
    ranges: list[tuple[int, int]] = []
    for a in range(len(split_idx) - 1):
        lo = int(u[split_idx[a]])
        hi = int(u[split_idx[a + 1]])
        if lo <= hi:
            ranges.append((lo, hi))
    return ranges if ranges else [(dmin, dmax)]


def degree_bin_label(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else f"{lo}--{hi}"


def bin_for_degree(deg: int, ranges: list[tuple[int, int]]) -> str:
    for lo, hi in ranges:
        if lo <= deg <= hi:
            return degree_bin_label(lo, hi)
    return degree_bin_label(ranges[-1][0], ranges[-1][1])


def plot_star_hub_leaf(by_folder: dict, out_fig: str):
    data = by_folder.get("star", [])
    if not data:
        return
    # (chunk, role) -> list of seed-level means
    acc_store: dict[tuple[int, str], list[float]] = defaultdict(list)
    auc_store: dict[tuple[int, str], list[float]] = defaultdict(list)
    max_store: dict[tuple[int, str], list[float]] = defaultdict(list)

    for meta, rows in data:
        hub, leaves = star_hub_leaf(rows)
        ch = meta["chunk"]
        if hub:
            acc_store[(ch, "hub")].append(mean_metric(hub, "global_test_acc"))
            auc_store[(ch, "hub")].append(mean_metric(hub, "avg_auc"))
            max_store[(ch, "hub")].append(mean_metric(hub, "max_auc"))
        if leaves:
            acc_store[(ch, "leaves")].append(mean_metric(leaves, "global_test_acc"))
            auc_store[(ch, "leaves")].append(mean_metric(leaves, "avg_auc"))
            max_store[(ch, "leaves")].append(mean_metric(leaves, "max_auc"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    roles = ["hub", "leaves"]
    x = np.arange(len(roles))
    w = 0.2
    offsets = {"baseline": -w * 1.5, "chunk": w * 0.5}
    colors = {"baseline": "#2c5282", "chunk": "#c05621"}

    for ci, cond in enumerate(["baseline", "chunk"]):
        ch = 0 if cond == "baseline" else 1
        acc_m, acc_e, leak_m, leak_e = [], [], [], []
        for role in roles:
            am, ae = ms(acc_store.get((ch, role), []))
            xm, xe = ms(max_store.get((ch, role), []))
            acc_m.append(am)
            acc_e.append(ae)
            leak_m.append(xm)
            leak_e.append(xe)
        off = offsets[cond]
        lbl = "Baseline" if cond == "baseline" else "Chunking"
        ax1.bar(x + off, acc_m, w, yerr=acc_e, label=lbl, color=colors[cond], capsize=3, ecolor="#333")
        ax2.bar(x + off, leak_m, w, yerr=leak_e, label=lbl, color=colors[cond], capsize=3, ecolor="#333")

    ax1.set_xticks(x)
    ax1.set_xticklabels(["Hub", "Leaves"])
    ax1.set_ylabel(YLABEL_TOP5_ACC)
    ax1.set_ylim(0, 0.65)
    ax1.legend()
    ax1.set_title("(a) Utility")

    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Hub", "Leaves"])
    ax2.set_ylabel(YLABEL_MIA_MAX)
    ax2.set_ylim(0.45, 1.05)
    ax2.legend()
    ax2.set_title("(b) Membership leakage")

    plt.tight_layout()
    save_both(fig, out_fig, "topology_star_hub_leaf")
    plt.close()

    return acc_store, auc_store, max_store


def plot_grid_roles(by_folder: dict, out_fig: str):
    data = by_folder.get("grid", [])
    if not data:
        return
    deg_labels = {2: "Corners ($d{=}2$)", 3: "Edges ($d{=}3$)", 4: "Interior ($d{=}4$)"}
    degs = [2, 3, 4]
    store_max: dict[tuple[int, int], list[float]] = defaultdict(list)
    store_acc: dict[tuple[int, int], list[float]] = defaultdict(list)

    for meta, rows in data:
        ch = meta["chunk"]
        for d in degs:
            sub = [r for r in rows if int(r["degree"]) == d]
            if not sub:
                continue
            store_acc[(ch, d)].append(mean_metric(sub, "global_test_acc"))
            store_max[(ch, d)].append(mean_metric(sub, "max_auc"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))
    x = np.arange(len(degs))
    w = 0.36
    for ch, name, color in ((0, "Baseline", "#2c5282"), (1, "Chunking", "#c05621")):
        am = [ms(store_acc.get((ch, d), []))[0] for d in degs]
        ae = [ms(store_acc.get((ch, d), []))[1] for d in degs]
        um = [ms(store_max.get((ch, d), []))[0] for d in degs]
        ue = [ms(store_max.get((ch, d), []))[1] for d in degs]
        off = -w / 2 if ch == 0 else w / 2
        ax1.bar(x + off, am, w, yerr=ae, label=name, color=color, capsize=3)
        ax2.bar(x + off, um, w, yerr=ue, label=name, color=color, capsize=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels([deg_labels[d] for d in degs], rotation=10, ha="right")
    ax1.set_ylabel(YLABEL_TOP5_ACC)
    ax1.set_title("(a) Utility by spatial role")
    ax1.legend()
    ax1.set_ylim(0.35, 0.55)

    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels([deg_labels[d] for d in degs], rotation=10, ha="right")
    ax2.set_ylabel(YLABEL_MIA_MAX)
    ax2.set_title("(b) Leakage by spatial role")
    ax2.legend()
    ax2.set_ylim(0.45, 1.02)

    plt.tight_layout()
    save_both(fig, out_fig, "topology_grid_degree_roles")
    plt.close()


def _bin_sort_key(lbl: str) -> tuple[int, int]:
    if "--" in lbl:
        a, b = lbl.split("--", 1)
        return (int(a), int(b))
    return (int(lbl), int(lbl))


def _degree_to_bin_label(deg: int, er_folder: str) -> str | None:
    spec = ER_DEGREE_BIN_SPEC.get(er_folder)
    if spec is None:
        return None
    for e in spec:
        if _fixed_er_bin_match(deg, e):
            return e["label"]
    return None


def _er_bin_stores(
    data: list[tuple[dict, list[dict]]], er_folder: str
) -> tuple[dict, dict, dict, list[str]]:
    """Per-seed means within degree bins (fixed spec per folder, else equal-frequency on pooled degrees)."""
    pooled = pooled_degrees(data)
    fixed = ER_DEGREE_BIN_SPEC.get(er_folder)
    if fixed is None:
        ranges = equal_frequency_degree_bin_ranges(pooled, k=TARGET_ER_BINS)

        def label_for_deg(deg: int) -> str:
            return bin_for_degree(deg, ranges)

        bin_order = [degree_bin_label(lo, hi) for lo, hi in ranges]
    else:
        bin_order = [e["label"] for e in fixed]

        def label_for_deg(deg: int) -> str:
            return _degree_to_bin_label(deg, er_folder) or ""

    store_acc: dict[tuple[int, str], list[float]] = defaultdict(list)
    store_auc: dict[tuple[int, str], list[float]] = defaultdict(list)
    store_max: dict[tuple[int, str], list[float]] = defaultdict(list)

    for meta, rows in data:
        ch = meta["chunk"]
        by_bin: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            b = label_for_deg(int(r["degree"]))
            if not b:
                continue
            by_bin[b].append(r)
        for b, sub in by_bin.items():
            if len(sub) < MIN_NODES_IN_BIN_PER_RUN:
                continue
            store_acc[(ch, b)].append(mean_metric(sub, "global_test_acc"))
            store_auc[(ch, b)].append(mean_metric(sub, "avg_auc"))
            store_max[(ch, b)].append(mean_metric(sub, "max_auc"))

    seen = {b for (ch, b) in store_acc.keys()} | {b for (ch, b) in store_max.keys()}
    if fixed is not None:
        active = [lbl for lbl in bin_order if lbl in seen]
    else:
        active = sorted(seen, key=_bin_sort_key)
    return store_acc, store_auc, store_max, active


def plot_er_bins_faceted(by_folder: dict, out_fig: str, er_folders: list[str]):
    """One row per ER $p$: left = accuracy by degree bin, right = MIA AUC by degree bin."""
    er_folders = sorted(er_folders, key=lambda f: float(f.replace("er_p_", "")))
    if not er_folders:
        return
    n = len(er_folders)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.85 * n), sharex=False)
    if n == 1:
        axes = np.array([axes])
    w = 0.36

    for i, er_folder in enumerate(er_folders):
        data = by_folder.get(er_folder, [])
        if not data:
            continue
        store_acc, _, store_max, active = _er_bin_stores(data, er_folder)
        if not active:
            continue
        ax1, ax2 = axes[i, 0], axes[i, 1]
        x = np.arange(len(active))
        ptag = er_folder.replace("er_p_", "")
        for ch, name, color in ((0, "Baseline", "#2c5282"), (1, "Chunking", "#c05621")):
            am = [ms(store_acc.get((ch, b), []))[0] for b in active]
            ae = [ms(store_acc.get((ch, b), []))[1] for b in active]
            um = [ms(store_max.get((ch, b), []))[0] for b in active]
            ue = [ms(store_max.get((ch, b), []))[1] for b in active]
            off = -w / 2 if ch == 0 else w / 2
            ax1.bar(x + off, am, w, yerr=ae, label=name, color=color, capsize=3)
            ax2.bar(x + off, um, w, yerr=ue, label=name, color=color, capsize=3)

        ax1.set_xticks(x)
        ax1.set_xticklabels(active, fontsize=9)
        ax1.set_ylabel(YLABEL_TOP5_ACC, fontsize=10)
        if i == 0:
            ax1.legend(fontsize=8, loc="upper right")
        ax1.set_title(f"(a) Utility  ($p={ptag}$)", fontsize=10)
        ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax2.set_xticks(x)
        ax2.set_xticklabels(active, fontsize=9)
        ax2.set_ylabel(YLABEL_MIA_MAX, fontsize=10)
        if i == 0:
            ax2.legend(fontsize=8, loc="upper right")
        ax2.set_title(f"(b) Leakage  ($p={ptag}$)", fontsize=10)
        ax2.set_ylim(0.45, 1.02)

    plt.tight_layout()
    save_both(fig, out_fig, "topology_er_degree_bins")
    plt.close()


def plot_violin_star_ring(by_folder: dict, out_fig: str):
    """Per-node max_auc at last round, pooled over seeds — star vs ring, baseline vs chunk."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    topo_names = [("star", "Star"), ("ring", "Ring")]
    conditions = [(0, "Baseline"), (1, "Chunking")]
    colors = {"Baseline": "#2c5282", "Chunking": "#c05621"}

    for ax, (folder, title) in zip(axes, topo_names):
        data = by_folder.get(folder, [])
        positions = []
        violins = []
        labels = []
        pos = 1
        for ch, cname in conditions:
            vals = []
            for meta, rows in data:
                if meta["chunk"] != ch:
                    continue
                for r in rows:
                    v = _fnum(r["max_auc"])
                    if v == v:
                        vals.append(v)
            if vals:
                violins.append(vals)
                positions.append(pos)
                labels.append(cname)
                pos += 1.2
        if violins:
            parts = ax.violinplot(
                violins,
                positions=positions,
                widths=0.55,
                showmeans=True,
                showmedians=False,
                showextrema=True,
            )
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors[labels[i]])
                pc.set_alpha(0.65)
            for key in ("cbars", "cmins", "cmaxes", "cmeans"):
                if key in parts:
                    parts[key].set_color("#333")
                    parts[key].set_linewidth(1)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylim(0.45, 1.02)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel(YLABEL_MIA_MAX_PER_NODE)

    plt.tight_layout()
    save_both(fig, out_fig, "topology_star_ring_auc_violin")
    plt.close()


def fmt(mu: float, sd: float, digits: int = 3) -> str:
    if mu != mu:
        return "---"
    if sd != sd or sd == 0:
        return f"{mu:.{digits}f}"
    return f"{mu:.{digits}f} $\\pm$ {sd:.{digits}f}"


def write_star_table(path: str, acc_s, auc_s, max_s):
    """acc_s etc.: dict (chunk, role) -> list of seed means."""
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Star topology: hub (degree $N{-}1$) vs.\ leaves (degree $1$). "
        r"Values are mean $\pm$ std across five random seeds; within each seed, metrics are averaged over nodes in the role.}",
        r"  \label{tab:star_hub_leaf}",
        r"  \begin{tabular}{@{}l l ccc@{}}",
        r"    \toprule",
        r"    Mode & Role & Accuracy & MIA AUC (avg) & MIA AUC (max mean)\tabularnewline",
        r"    \midrule",
    ]
    for cond, ch in [("Baseline", 0), ("Chunking", 1)]:
        for role, rlab in [("hub", "Hub"), ("leaves", "Leaves")]:
            am, ae = ms(acc_s.get((ch, role), []))
            um, ue = ms(auc_s.get((ch, role), []))
            xm, xe = ms(max_s.get((ch, role), []))
            lines.append(
                f"    {cond} & {rlab} & {fmt(am, ae)} & {fmt(um, ue)} & {fmt(xm, xe)}\\tabularnewline"
            )
        if cond == "Baseline":
            lines.append(r"    \midrule")
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def write_grid_table(path: str, by_folder: dict):
    data = by_folder.get("grid", [])
    rows_tex = []
    degs = [2, 3, 4]
    roles = {2: "Corners ($d{=}2$)", 3: "Edges ($d{=}3$)", 4: "Interior ($d{=}4$)"}
    store_acc: dict[tuple[int, int], list[float]] = defaultdict(list)
    store_auc: dict[tuple[int, int], list[float]] = defaultdict(list)
    store_max: dict[tuple[int, int], list[float]] = defaultdict(list)
    for meta, rows in data:
        ch = meta["chunk"]
        for d in degs:
            sub = [r for r in rows if int(r["degree"]) == d]
            if not sub:
                continue
            store_acc[(ch, d)].append(mean_metric(sub, "global_test_acc"))
            store_auc[(ch, d)].append(mean_metric(sub, "avg_auc"))
            store_max[(ch, d)].append(mean_metric(sub, "max_auc"))

    for cond, ch in [("Baseline", 0), ("Chunking", 1)]:
        for d in degs:
            am, ae = ms(store_acc.get((ch, d), []))
            um, ue = ms(store_auc.get((ch, d), []))
            xm, xe = ms(store_max.get((ch, d), []))
            rows_tex.append(
                f"    {cond} & {roles[d]} & {fmt(am, ae)} & {fmt(um, ue)} & {fmt(xm, xe)}\\tabularnewline"
            )
        if cond == "Baseline":
            rows_tex.append(r"    \midrule")

    content = "\n".join(
        [
            r"\begin{table}[t]",
            r"  \centering",
            r"  \caption{Grid topology: nodes grouped by graph degree (corner / edge / interior on the $10{\times}10$ grid). "
            r"Mean $\pm$ std across five seeds; each seed value averages over nodes in that degree class.}",
            r"  \label{tab:grid_degree_roles}",
            r"  \begin{tabular}{@{}l l ccc@{}}",
            r"    \toprule",
            r"    Mode & Role & Accuracy & MIA AUC (avg) & MIA AUC (max mean)\tabularnewline",
            r"    \midrule",
            *rows_tex,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content + "\n")
    print(f"Wrote {path}")


def write_er_tables_all(path: str, by_folder: dict, er_dirs: list[str]):
    r"""Single table* \label{tab:er_degree_bins_all}: one block per $p$ with \multirow."""
    body_lines: list[str] = []
    blocks_out = 0
    for er_folder in sorted(er_dirs, key=lambda f: float(f.replace("er_p_", ""))):
        data = by_folder.get(er_folder, [])
        store_acc, store_auc, store_max, active = _er_bin_stores(data, er_folder)
        if not active:
            continue
        ptag = er_folder.replace("er_p_", "")
        row_entries: list[tuple[str, str, str, str, str]] = []
        for cond, ch in [("Baseline", 0), ("Chunking", 1)]:
            for b in active:
                am, ae = ms(store_acc.get((ch, b), []))
                um, ue = ms(store_auc.get((ch, b), []))
                xm, xe = ms(store_max.get((ch, b), []))
                row_entries.append(
                    (cond, b, fmt(am, ae), fmt(um, ue), fmt(xm, xe))
                )
        nrows = len(row_entries)
        blocks_out += 1
        if blocks_out > 1:
            body_lines.append(r"\midrule")
            body_lines.append("")
        for i, (cond, b, acc_s, auc_a, auc_m) in enumerate(row_entries):
            if i == 0:
                body_lines.append(
                    rf"\multirow{{{nrows}}}{{*}}{{{ptag}}} & {cond} & {b} & {acc_s} & {auc_a} & {auc_m} \\"
                )
            else:
                body_lines.append(rf"& {cond} & {b} & {acc_s} & {auc_a} & {auc_m} \\")

    if not body_lines:
        body_lines.append(r"% (no ER degree-bin data)")

    content = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Erd\H{o}s--R\'enyi graphs: node groups by degree bin for different edge probabilities $p$. "
            r"Each entry averages over nodes in the bin. Bins with fewer than three nodes in a run are omitted.}",
            r"\label{tab:er_degree_bins_all}",
            r"\begin{tabular}{@{}c l l ccc@{}}",
            r"\toprule",
            r"$p$ & Mode & Degree bin & Accuracy & AUC$_{\text{avg}}$ & AUC$_{\text{max}}$ \\",
            r"\midrule",
            "",
            "\n".join(body_lines),
            r"",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content + "\n")
    print(f"Wrote {path} ({blocks_out} probability block(s))")


def write_homogeneous_table(path: str, by_folder: dict):
    """Max MIA AUC: network mean per run (col 3) and within-run SD of per-node max (col 4); both mean±std across seeds."""
    folders = [
        ("ring", "Ring"),
        ("full", "Fully conn."),
        ("regular_d3", r"$3$-regular"),
        ("regular_d10", r"$10$-regular"),
        ("regular_d25", r"$25$-regular"),
    ]
    rows_tex = []
    for fid, name in folders:
        for row_idx, (cond, ch) in enumerate([("Baseline", 0), ("Chunking", 1)]):
            within_spreads = []
            net_means = []
            for meta, rows in by_folder.get(fid, []):
                if meta["chunk"] != ch:
                    continue
                vals = [_fnum(r["max_auc"]) for r in rows]
                vals = [a for a in vals if a == a]
                if len(vals) > 1:
                    within_spreads.append(pstdev(vals))
                elif len(vals) == 1:
                    within_spreads.append(0.0)
                net_means.append(mean(vals) if vals else float("nan"))
            sm, ss = ms(within_spreads)
            mm, me = ms(net_means)
            topo = name if row_idx == 0 else ""
            rows_tex.append(
                f"    {topo} & {cond} & {fmt(mm, me)} & {fmt(sm, ss)} \\\\"
            )
        rows_tex.append(r"    \midrule")
    if rows_tex and rows_tex[-1].endswith(r"\midrule"):
        rows_tex.pop()

    content = "\n".join(
        [
            r"\begin{table}[t]",
            r"  \centering",
            r"  \caption{Graphs with homogeneous node distribution using per-node \emph{max} MIA AUC. \textbf{Column~3:} in each run, mean $\mathrm{MIA}^{\max}$ over peers; reported value is mean $\pm$ \emph{std across random seeds} (seed-to-seed variability of the network mean). \textbf{Column~4:} in each run, population SD of $\mathrm{MIA}^{\max}$ across the $N$ peers; reported value is mean $\pm$ \emph{std across seeds} of that within-run spread (how peer-level spread itself shifts from seed to seed).}",
            r"  \label{tab:homogeneous_spread}",
            r"  \scriptsize",
            r"  \setlength{\tabcolsep}{3pt}",
            r"  \resizebox{\columnwidth}{!}{%",
            r"  \begin{tabular}{@{}l l cc@{}}",
            r"    \toprule",
            r"    Topology & Mode & Net mean $\mathrm{MIA}^{\max}$ & Within-run SD (nodes) \\",
            r"    \midrule",
            *rows_tex,
            r"    \bottomrule",
            r"  \end{tabular}%",
            r"  }",
            r"\end{table}",
        ]
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content + "\n")
    print(f"Wrote {path}")


def write_summary_network_table(path: str, by_folder: dict):
    """Aggregate network-level (all nodes) mean per seed — full sweep summary."""
    order = [
        "ring",
        "star",
        "grid",
        "full",
        "regular_d3",
        "regular_d10",
        "regular_d25",
    ]
    er_fs = sorted(f for f in by_folder if f.startswith("er_p_"))
    order.extend(er_fs)

    display = {
        "ring": "Ring",
        "star": "Star",
        "grid": "Grid",
        "full": "Fully connected",
        "regular_d3": r"$3$-regular",
        "regular_d10": r"$10$-regular",
        "regular_d25": r"$25$-regular",
    }

    def er_label(fid: str) -> str:
        if fid.startswith("er_p_"):
            p = fid.replace("er_p_", "")
            return f"ER ($p={p}$)"
        return fid


    rows_tex = []
    for fid in order:
        label = display.get(fid, er_label(fid) if fid.startswith("er_p_") else fid.replace("_", " "))
        for row_idx, (cond, ch) in enumerate([("Baseline", 0), ("Chunking", 1)]):
            accs, max_aucs, avg_aucs = [], [], []
            for meta, rows in by_folder.get(fid, []):
                if meta["chunk"] != ch:
                    continue
                accs.append(mean_metric(rows, "global_test_acc"))
                max_aucs.append(mean_metric(rows, "max_auc"))
                avg_aucs.append(mean_metric(rows, "avg_auc"))
            am, ae = ms(accs)
            umax_m, umax_e = ms(max_aucs)
            uavg_m, uavg_e = ms(avg_aucs)
            topo = label if row_idx == 0 else ""
            rows_tex.append(
                f"    {topo} & {cond} & {fmt(am, ae)} & {fmt(umax_m, umax_e)} & {fmt(uavg_m, uavg_e)} \\\\"
            )
        rows_tex.append(r"    \midrule")
    if rows_tex[-1].endswith(r"\midrule"):
        rows_tex.pop()

    content = "\n".join(
        [
            r"\begin{table*}[t]",
            r"  \centering",
            r"  \caption{Network-level summary: for each topology and training mode, global test accuracy and mean node-level MIA AUC (\emph{max} and \emph{avg}), each averaged over all $N$ peers in the final round, then reported as mean $\pm$ std across seeds.}",
            r"  \label{tab:network_level_summary}",
            r"  \small",
            r"  \begin{tabular}{@{}l l ccc@{}}",
            r"    \toprule",
            r"    Topology & Mode & Accuracy & AUC$_{\text{max}}$ & AUC$_{\text{avg}}$ \\",
            r"    \midrule",
            *rows_tex,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table*}",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content + "\n")
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="synced/results/cifar100/baseline_chunk_topology_sweep")
    ap.add_argument("--out-dir-figures", type=str, default="Master_Thesis/figures")
    ap.add_argument("--out-dir-tables", type=str, default="Master_Thesis/tables")
    args = ap.parse_args()

    thesis_style()
    by_folder = collect_by_folder(args.root)

    acc_s, auc_s, max_s = {}, {}, {}
    r = plot_star_hub_leaf(by_folder, args.out_dir_figures)
    if r:
        acc_s, auc_s, max_s = r

    plot_grid_roles(by_folder, args.out_dir_figures)
    er_dirs = sorted(f for f in by_folder if f.startswith("er_p_"))
    if er_dirs:
        plot_er_bins_faceted(by_folder, args.out_dir_figures, er_dirs)
    plot_violin_star_ring(by_folder, args.out_dir_figures)

    tdir = args.out_dir_tables
    if acc_s:
        write_star_table(os.path.join(tdir, "table_star_hub_leaf.tex"), acc_s, auc_s, max_s)
    write_grid_table(os.path.join(tdir, "table_grid_degree_roles.tex"), by_folder)
    if er_dirs:
        write_er_tables_all(os.path.join(tdir, "table_er_degree_bins.tex"), by_folder, er_dirs)
    write_homogeneous_table(os.path.join(tdir, "table_homogeneous_spread.tex"), by_folder)
    write_summary_network_table(os.path.join(tdir, "table_network_level_summary.tex"), by_folder)


if __name__ == "__main__":
    main()
