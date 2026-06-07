#!/usr/bin/env python3
"""
Draw communication topology graphs for the thesis (PDFs).
Usage: from project root, run
    python3 -m plotting.misc.draw_thesis_topologies
Output: plots/misc/ring.pdf, fully_connected.pdf, er.pdf, star.pdf, grid.pdf, regular.pdf
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.topologies.topology_factory import TopologyFactory

OUT_DIR = _REPO_ROOT / "plots" / "misc"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Small illustrative graphs: 8--12 nodes
N = 10
GRID_N = 12  # 3x4 grid
SEED = 42

def draw_and_save(topology, filename: str):
    path = OUT_DIR / filename
    topology.draw(title=None, save_path=str(path))
    plt.close("all")
    print(f"  {path}")


def main():
    print("Writing topology PDFs to", OUT_DIR)

    # Ring
    t = TopologyFactory.create("ring", N)
    draw_and_save(t, "ring.pdf")

    # Fully connected
    t = TopologyFactory.create("full", N)
    draw_and_save(t, "fully_connected.pdf")

    # Erdős–Rényi (p so that graph is likely connected but not dense)
    t = TopologyFactory.create("er", N, p=0.25, seed=SEED)
    draw_and_save(t, "er.pdf")

    # Star
    t = TopologyFactory.create("star", N)
    draw_and_save(t, "star.pdf")

    # Grid (e.g. 3x4)
    t = TopologyFactory.create("grid", GRID_N)
    draw_and_save(t, "grid.pdf")

    # Random d-regular (degree 3; 10*3 even)
    t = TopologyFactory.create("regular", N, degree=3, seed=SEED)
    draw_and_save(t, "regular.pdf")

if __name__ == "__main__":
    main()
