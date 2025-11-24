# star_topology.py

import networkx as nx
import matplotlib.pyplot as plt

from .topology import Topology


class StarTopology(Topology):
    """
    Star topology:
    - Node 0 is the hub.
    - All other nodes connect only to node 0.
    """

    def __init__(self, num_nodes: int, shuffle_nodes: bool = False):
        super().__init__(num_nodes, shuffle_nodes=shuffle_nodes)

    def _build_graph(self) -> nx.Graph:
        # NetworkX creates: 0 -- {1..N-1}
        return nx.star_graph(self.num_nodes - 1)

    def draw(self, title: str | None = None, save_path: str | None = None):
        # Use a shell layout to highlight the center node
        pos = nx.shell_layout(self.graph, nlist=[[0], list(range(1, self.num_nodes))])
        plt.figure(figsize=(4, 4))

        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=600,
            font_size=10,
            width=0.7,
        )

        plt.title(title or "Star topology")
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
        # plt.show()
