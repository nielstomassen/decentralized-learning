# topology_factory.py

from typing import Any

from src.topologies import (
    Topology,
    RingTopology,
    FullyConnectedTopology,
    ErdosRenyiTopology,
    SmallWorldTopology,
)


class TopologyFactory:
    @staticmethod
    def create(name: str, num_nodes: int, shuffle_nodes: bool = False, **kwargs: Any) -> Topology:
        """
        Create a topology instance by name.

        Examples:
            TopologyFactory.create("ring", 10, shuffle_nodes=True)
            TopologyFactory.create("fully", 10)
            TopologyFactory.create("er", 10, p=0.2)
            TopologyFactory.create("small_world", 10, k=4, p=0.1)
        """
        name = name.lower()

        if name in ("ring", "cycle"):
            return RingTopology(num_nodes, shuffle_nodes=shuffle_nodes)

        if name in ("full", "fully_connected", "clique"):
            return FullyConnectedTopology(num_nodes, shuffle_nodes=shuffle_nodes)

        if name in ("er", "erdos_renyi"):
            p = kwargs.get("p", 0.2)
            return ErdosRenyiTopology(num_nodes, p=p, shuffle_nodes=shuffle_nodes)

        if name in ("small_world", "ws"):
            k = kwargs.get("k", 4)
            p = kwargs.get("p", 0.1)
            return SmallWorldTopology(num_nodes, k=k, p=p, shuffle_nodes=shuffle_nodes)

        raise ValueError(f"Unknown topology: {name}")
