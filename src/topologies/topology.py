# topology.py

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import random
import networkx as nx

class Topology(ABC):
    def __init__(self, num_nodes: int, shuffle_nodes: bool = False):
        self.num_nodes = num_nodes

        g = self._build_graph()

        if shuffle_nodes:
            g = self._shuffle_graph_nodes(g)

        self.graph = g
        self.neighbors = self._graph_to_neighbors(self.graph)

    @abstractmethod
    def _build_graph(self) -> nx.Graph:
        """Subclasses implement this to construct the NetworkX graph."""
        pass

    def _graph_to_neighbors(self, g: nx.Graph) -> Dict[int, List[int]]:
        return {int(n): [int(nb) for nb in g.neighbors(n)] for n in g.nodes()}

    def _shuffle_graph_nodes(self, g: nx.Graph) -> nx.Graph:
        """
        Randomly permute node labels.
        Reproducible if you call set_global_seed(...) earlier.
        """
        nodes = list(g.nodes())
        permuted = nodes[:]
        random.shuffle(permuted)  # uses global random seed
        mapping = dict(zip(nodes, permuted))
        return nx.relabel_nodes(g, mapping)

    @abstractmethod
    def draw(self, title: str | None = None, save_path: str | None = None):
        """Draw this topology with a nice layout."""
        pass
