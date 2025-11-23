from typing import Optional
import networkx as nx
import matplotlib.pyplot as plt
from .topology import Topology

class ErdosRenyiTopology(Topology):
    def __init__(self, num_nodes: int, p: float, shuffle_nodes: bool = False):
        self.p = p
        super().__init__(num_nodes, shuffle_nodes=shuffle_nodes)

    def _build_graph(self) -> nx.Graph:
        return nx.erdos_renyi_graph(self.num_nodes, self.p)

    def draw(self, title: str | None = None, save_path: str | None = None):
        pos = nx.spring_layout(self.graph, seed=0)
        plt.figure(figsize=(4, 4))
        nx.draw(self.graph, pos, with_labels=True, node_size=600, font_size=10, width=1.0)
        
        plt.title(title or f"ER topology (p={self.p})")
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
        # plt.show()
