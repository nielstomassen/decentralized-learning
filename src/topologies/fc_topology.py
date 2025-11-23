from typing import Optional
import networkx as nx
import matplotlib.pyplot as plt
from .topology import Topology

class FullyConnectedTopology(Topology):
    def __init__(self, num_nodes: int, shuffle_nodes: bool = False):
        super().__init__(num_nodes, shuffle_nodes=shuffle_nodes)

    def _build_graph(self) -> nx.Graph:
        return nx.complete_graph(self.num_nodes)

    def draw(self, title: str | None = None, save_path: str | None = None):
        pos = nx.spring_layout(self.graph, seed=0)
        plt.figure(figsize=(4, 4))
        nx.draw(self.graph, pos, with_labels=True, node_size=600, font_size=10, width=0.7)
        
        plt.title(title or "Fully connected topology")
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
        # plt.show()

