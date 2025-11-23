from typing import Optional
import networkx as nx
import matplotlib.pyplot as plt
from .topology import Topology

class RingTopology(Topology):
    def __init__(self, num_nodes: int, shuffle_nodes: bool = False):
        super().__init__(num_nodes, shuffle_nodes=shuffle_nodes)

    def _build_graph(self) -> nx.Graph:
        return nx.cycle_graph(self.num_nodes)

    def draw(self, title: str | None = None, save_path: str | None = None):
        pos = nx.circular_layout(self.graph)
        plt.figure(figsize=(4, 4))
        nx.draw(self.graph, pos, with_labels=True, node_size=600, font_size=10, width=1.5)
      
        plt.title(title or "Ring topology")
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
        # plt.show()