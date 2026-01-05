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
        self.weights = self._build_metropolis_weights()

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
    
    def _build_metropolis_weights(self):
        g = self.graph
        degrees = dict(g.degree())
        weights = {int(i): {} for i in g.nodes()}  # row i → dict(neighbor->w_ij)

        # Off-diagonal weights for undirected edges
        for i, j in g.edges():
            deg_i = degrees[i]
            deg_j = degrees[j]
            w_ij = 1.0 / max(deg_i, deg_j)
            # symmetric
            weights[int(i)][int(j)] = w_ij
            weights[int(j)][int(i)] = w_ij

        # Diagonal weights
        for i in g.nodes():
            row = weights[int(i)]
            s_off = sum(row.values())
            row[int(i)] = 1.0 - s_off  # self weight

        return weights

    @abstractmethod
    def draw(self, title: str | None = None, save_path: str | None = None):
        """Draw this topology with a nice layout."""
        pass

