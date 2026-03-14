#!/usr/bin/env python3
"""
Chunking schematic for thesis: sender splits each tensor into d chunks;
chunk order permuted per row (no fixed column→neighbor); one receiver
gets same color (same chunk index) to show random assignment.
Usage: python scripts/draw_chunking_schematic.py
Output: Master_Thesis/figures/chunking_schematic.pdf
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.join("Master_Thesis", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

NUM_TENSORS = 3
D = 3
# assign[j][ti] = chunk index (0,1,2) that neighbor j gets from tensor ti.
# N1 gets same chunk index from every tensor → same color in receiver (random assignment).
# N0 and N2 middle blocks correct (not swapped): N0 middle=0, N2 middle=2.
ASSIGN = [
    [0, 0, 0],   # N0: mixed (middle = 0)
    [1, 1, 1],   # N1: same color
    [2, 2, 2],   # N2: mixed (middle = 2)
]
# Display order of chunks per row (so columns don't = neighbors). Row ti: chunk at position p is ORDER[ti][p].
ORDER = [
    [0, 1, 2],   # T1: a b c
    [1, 2, 0],   # T2: b c a
    [2, 0, 1],   # T3: c a b
]


def main():
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.6))
    ax.set_aspect("equal")
    ax.axis("off")

    # Set2 colormap (same style as first chunking schematic / topology plots)
    colors = [plt.cm.Set2(i) for i in (0.2, 0.45, 0.7)]
    left = 0.2
    sender_w = 2.0
    sender_h = 1.35
    chunk_w = (sender_w - 0.3) / D
    row_h = 0.32
    row_gap = 0.05
    y0 = 0.25 + sender_h - row_h - 0.12

    # Sender box
    ax.add_patch(mpatches.FancyBboxPatch(
        (left, 0.25), sender_w, sender_h,
        boxstyle="round,pad=0.02", facecolor="lightgray", edgecolor="black", linewidth=1.2
    ))
    ax.text(left + sender_w / 2, 0.25 + sender_h + 0.08, "Sender", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Sender rows: chunk order permuted per row (ORDER) so no column = one neighbor
    for ti in range(NUM_TENSORS):
        y = y0 - ti * (row_h + row_gap)
        for p in range(D):
            c = ORDER[ti][p]
            ax.add_patch(mpatches.Rectangle(
                (left + 0.15 + p * chunk_w, y), chunk_w - 0.04, row_h - 0.02,
                facecolor=colors[c], edgecolor="black", linewidth=0.7
            ))
        ax.text(left - 0.05, y + (row_h - 0.02) / 2, f"T{ti+1}", ha="right", va="center", fontsize=9)

    # Receivers: vertically centered with sender (middle receiver aligned with middle sender row)
    rx = 3.8
    rw = 0.85
    rchunk_h = 0.12
    rh = 3 * rchunk_h + 0.14
    r_gap = 0.55
    sender_center_y = y0 - (row_h + row_gap) - (row_h - 0.02) / 2
    r_base = sender_center_y + r_gap

    for j in range(D):
        ry = r_base - j * r_gap
        ax.add_patch(mpatches.FancyBboxPatch(
            (rx, ry), rw, rh,
            boxstyle="round,pad=0.02", facecolor="white", edgecolor="black", linewidth=1
        ))
        for ti in range(NUM_TENSORS):
            c = ASSIGN[j][ti]
            cy = ry + rh - 0.09 - (ti + 1) * rchunk_h
            ax.add_patch(mpatches.Rectangle(
                (rx + 0.07, cy), rw - 0.14, rchunk_h - 0.02,
                facecolor=colors[c], edgecolor="black", linewidth=0.5
            ))
        ax.text(rx + rw + 0.06, ry + rh / 2, f"N{j+1}", ha="left", va="center", fontsize=10, fontweight="bold")

    # Arrows: one per neighbor, colored like that neighbor's blocks (use first block color for consistency)
    x_src = left + sender_w
    for j in range(D):
        ys = [y0 - ti * (row_h + row_gap) + (row_h - 0.02) / 2 for ti in range(NUM_TENSORS)]
        y_src = sum(ys) / len(ys)
        y_dst = r_base - j * r_gap + rh / 2
        arrow_color = colors[ASSIGN[j][0]]
        ax.annotate("", xy=(rx - 0.02, y_dst), xytext=(x_src, y_src),
                    arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5, shrinkA=2, shrinkB=2))

    # Caption: below the diagram so it doesn't overlap (lowest receiver is at r_base - 2*r_gap)
    caption_y = r_base - 2 * r_gap - 0.22
    ax.text((left + sender_w + rx + rw) / 2, caption_y,
            "Each tensor $\\rightarrow$ $d$ chunks. Each neighbor randomly receives $k$ chunks per tensor (here $k$=1).", ha="center", va="top", fontsize=9, style="italic")

    ax.set_xlim(-0.12, rx + rw + 0.25)
    ax.set_ylim(caption_y - 0.2, 0.25 + sender_h + 0.25)

    out_path = os.path.join(OUT_DIR, "chunking_schematic.pdf")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.5, format="pdf")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
