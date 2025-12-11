# random_regular_topology.py

from typing import Optional
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from .topology import Topology

def adjacency_spectral_gap(g: nx.Graph) -> float:
    """
    Spectral gap of the adjacency matrix: λ1 - λ2.
    For a connected d-regular graph, λ1 = d.
    """
    A = nx.to_numpy_array(g, dtype=float)
    eigvals = np.linalg.eigvalsh(A)
    eigvals_sorted = np.sort(eigvals)[::-1]  # descending
    return float(eigvals_sorted[0] - eigvals_sorted[1])

class RandomRegularTopology(Topology):
    """
    Random d-regular graph (each node has exactly `degree` neighbors).

    This is a classic way to get graphs with good expansion and
    relatively large spectral gaps (for the adjacency matrix).

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    degree : int
        Regular degree d. Must satisfy 0 <= d < num_nodes and d * num_nodes is even.
    shuffle_nodes : bool, default False
        Whether to randomly relabel node indices after creation.
    seed : Optional[int], default None
        Seed for NetworkX RNG (optional). For global reproducibility
        you can also use the global random seed instead.
    """

    def __init__(
        self,
        num_nodes: int,
        degree: int,
        shuffle_nodes: bool = False,
        seed: Optional[int] = None,
    ):
        self.degree = degree
        self.seed = seed
        super().__init__(num_nodes, shuffle_nodes=shuffle_nodes)    

    def _build_graph(self) -> nx.Graph:
        d = self.degree
        n = self.num_nodes

        # Basic feasibility checks for random_regular_graph
        if d < 0 or d >= n:
            raise ValueError(f"Degree must satisfy 0 <= d < n (got d={d}, n={n}).")
        if (d * n) % 2 != 0:
            raise ValueError(
                f"d * n must be even for a d-regular graph (got d={d}, n={n})."
            )

        g = nx.random_regular_graph(d, n, seed=self.seed)
        print(adjacency_spectral_gap(g))
        return g

    def draw(self, title: str | None = None, save_path: str | None = None):
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(4, 4))
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=600,
            font_size=10,
            width=1.0,
        )

        plt.title(title or f"Random regular topology (d={self.degree})")
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
        # plt.show() 



