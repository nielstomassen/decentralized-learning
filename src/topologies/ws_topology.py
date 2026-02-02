import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional
from .topology import Topology


class SmallWorldTopology(Topology):
    def __init__(
        self,
        num_nodes: int,
        k: int,
        p: float,
        shuffle_nodes: bool = False,
        seed: Optional[int] = None,
    ):
        self.k = k
        self.p = p
        self.seed = seed
        super().__init__(num_nodes, shuffle_nodes=shuffle_nodes)

    def _build_graph(self) -> nx.Graph:
        # Watts–Strogatz small-world graph (REPRODUCIBLE)
        return nx.watts_strogatz_graph(
            self.num_nodes,
            self.k,
            self.p,
            seed=self.seed,
        )

    def draw(self, title: str | None = None, save_path: str | None = None):
        # Use same seed for layout for stable plots
        pos = nx.spring_layout(self.graph, seed=self.seed)
        plt.figure(figsize=(4, 4))
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=600,
            font_size=10,
            width=1.0,
        )

        plt.title(title or f"Small-world (k={self.k}, p={self.p}, seed={self.seed})")
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
