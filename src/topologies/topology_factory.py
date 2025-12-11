# topology_factory.py

from typing import Any

from src.topologies import (
    Topology,
    RingTopology,
    FullyConnectedTopology,
    ErdosRenyiTopology,
    SmallWorldTopology,
    StarTopology,
    GridTopology,
    RandomRegularTopology,
    StochasticBlockTopology
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

        elif name in ("full", "fully_connected", "clique"):
            return FullyConnectedTopology(num_nodes, shuffle_nodes=shuffle_nodes)

        elif name in ("er", "erdos_renyi"):
            p = kwargs.get("p", 0.2)
            return ErdosRenyiTopology(num_nodes, p=p, shuffle_nodes=shuffle_nodes)

        elif name in ("small_world", "ws"):
            k = kwargs.get("k", 4)
            p = kwargs.get("p", 0.1)
            return SmallWorldTopology(num_nodes, k=k, p=p, shuffle_nodes=shuffle_nodes)
        
        elif name in ("star", "hub"):
            return StarTopology(num_nodes, shuffle_nodes=shuffle_nodes)
        
        elif name in ("grid", "mesh", "lattice"):
            return GridTopology(num_nodes, shuffle_nodes=shuffle_nodes)
        
        elif name in ("regular", "random_regular"):
            degree = kwargs.get("degree", 3)
            return RandomRegularTopology(
                num_nodes,
                degree=degree,
                shuffle_nodes=shuffle_nodes,
                seed=kwargs.get("seed", None),
            )
        
        elif name in ("sbm", "block", "community"):
            num_blocks = kwargs.get("num_blocks", 2)
            p_in = kwargs.get("p_in", 0.5)
            p_out = kwargs.get("p_out", 0.05)
            return StochasticBlockTopology(
                num_nodes,
                num_blocks=num_blocks,
                p_in=p_in,
                p_out=p_out,
                shuffle_nodes=shuffle_nodes,
            )

        else:
            raise ValueError(f"Unknown topology: {name}")
