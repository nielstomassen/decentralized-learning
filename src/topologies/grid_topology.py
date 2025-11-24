# grid_topology.py

from typing import Optional, Tuple
import networkx as nx
import matplotlib.pyplot as plt

from .topology import Topology

class GridTopology(Topology):
    """
    2D grid topology.

    Creates a grid of size (rows x cols) such that:
        num_nodes == rows * cols

    Node labels are remapped to 0..num_nodes-1
    in row-major order.

    Example for rows=2, cols=3:

        0 -- 1 -- 2
        |    |    |
        3 -- 4 -- 5
    """

    def __init__(self, num_nodes: int, shuffle_nodes: bool = False):
        self.rows, self.cols = self._infer_grid_shape(num_nodes)
        super().__init__(num_nodes, shuffle_nodes=shuffle_nodes)

    def _infer_grid_shape(self, num_nodes: int) -> Tuple[int, int]:
        """
        Pick a near-square grid automatically.
        e.g. num_nodes=12 -> (3,4)
        """
        import math

        # Best square-ish factorization
        root = int(math.sqrt(num_nodes))
        for r in range(root, 0, -1):
            if num_nodes % r == 0:
                return r, num_nodes // r

        # fallback (degenerate grid)
        return 1, num_nodes

    def _build_graph(self) -> nx.Graph:
        # Build a 2D grid graph using tuple coordinates
        g = nx.grid_2d_graph(self.rows, self.cols)

        # Relabel nodes to 0..N-1 in row-major order
        mapping = {}
        counter = 0
        for r in range(self.rows):
            for c in range(self.cols):
                mapping[(r, c)] = counter
                counter += 1

        g = nx.relabel_nodes(g, mapping)
        return g

    def draw(self, title: str | None = None, save_path: str | None = None):
        # Draw using the true grid positions for clarity
        pos = {}
        counter = 0
        for r in range(self.rows):
            for c in range(self.cols):
                pos[counter] = (c, -r)   # negative y so grid is upright
                counter += 1

        plt.figure(figsize=(5, 5))
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=500,
            font_size=10,
            width=0.8,
        )

        plt.title(title or f"Grid topology ({self.rows}x{self.cols})")
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
        # plt.show()
