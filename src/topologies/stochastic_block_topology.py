# stochastic_block_topology.py

from typing import List, Optional
import networkx as nx
import matplotlib.pyplot as plt

from .topology import Topology

import numpy as np


def spectral_gap(g: nx.Graph) -> float:
    """
    Returns the algebraic connectivity (second smallest eigenvalue) of the
    normalized Laplacian, which is a standard 'spectral gap' measure.
    """
    L = nx.normalized_laplacian_matrix(g).astype(float).toarray()
    eigvals = np.linalg.eigvalsh(L)
    eigvals = sorted(eigvals)
    return eigvals[1]  # λ2


class StochasticBlockTopology(Topology):
    """
    Stochastic Block Model (SBM) topology with tunable community structure.
    """

    def __init__(
        self,
        num_nodes: int,
        num_blocks: int = 2,
        p_in: float = 0.5,
        p_out: float = 0.05,
        shuffle_nodes: bool = False,
        seed: Optional[int] = None,
    ):
        self.num_blocks = num_blocks
        self.p_in = p_in
        self.p_out = p_out
        self.seed = seed

        self._sizes: List[int] = []
        self._block_membership: List[int] = []

        super().__init__(num_nodes, shuffle_nodes=shuffle_nodes)

    def _build_graph(self) -> nx.Graph:
        # Split num_nodes as evenly as possible over `num_blocks`
        base = self.num_nodes // self.num_blocks
        sizes = [base] * self.num_blocks
        remainder = self.num_nodes - base * self.num_blocks
        for i in range(remainder):
            sizes[i] += 1

        self._sizes = sizes

        # Probability matrix: diagonal = p_in, off-diagonal = p_out
        probs = [[self.p_out for _ in range(self.num_blocks)]
                 for _ in range(self.num_blocks)]
        for i in range(self.num_blocks):
            probs[i][i] = self.p_in

        g = nx.stochastic_block_model(
            sizes,
            probs,
            seed=self.seed,
            directed=False,
        )

        # Block membership per node (in the original block-model labeling)
        membership: List[int] = []
        for block_idx, size in enumerate(sizes):
            membership.extend([block_idx] * size)
        self._block_membership = membership

        # Optional: don't print in library code (can be noisy), but keep if you want
        # print(spectral_gap(g))

        return g

    def draw(self, title: str | None = None, save_path: str | None = None):
        # Color nodes by block membership if we have it
        if self._block_membership and len(self._block_membership) == self.num_nodes:
            colors = [self._block_membership[int(n)] for n in self.graph.nodes()]
        else:
            colors = None

        pos = nx.spring_layout(self.graph, seed=self.seed)
        plt.figure(figsize=(4, 4))
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=600,
            font_size=10,
            width=1.0,
            node_color=colors,
            cmap=plt.cm.tab10 if colors is not None else None,
        )

        plt.title(
            title
            or f"Stochastic Block Topology (p_in={self.p_in}, p_out={self.p_out}, seed={self.seed})"
        )
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
