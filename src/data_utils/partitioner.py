from torch.utils.data import DataLoader, Subset
import numpy as np
import torch

def partition_dataset(dataset, num_nodes, batch_size=32, shuffle=True):
    # Create array [0, 1, ..., len(dataset) -1]
    indices = np.arange(len(dataset))
    if shuffle:
        # Randomly shuffle the array (seeded)
        np.random.shuffle(indices)

    # Create array of with split sizes for each node
    split_sizes = [len(indices) // num_nodes] * num_nodes
    # If not dividable by num_nodes add 1 to first n (remainder) split sizes
    # to ensure no samples are lost.
    for i in range(len(indices) % num_nodes):
        split_sizes[i] += 1

    subsets = []
    start = 0
    # Cut dataset into chunks
    for node_id, size in enumerate(split_sizes):
        end = start + size

        # Each node gets its own generator, but deterministic from `seed`
        g = torch.Generator()
        g.manual_seed(torch.initial_seed() + node_id)  # different per node, but reproducible

        # get current indices for this node
        subset_indices = indices[start:end]
        # Subset(...) creates a dataset containing only those specific indices
        # DataLoader(...) wraps that subset and:
        #   - iterates over it in mini-batches of size `batch_size`
        #   - shuffles the order each epoch (shuffle=True)
        #   - is used to loop over as: `for x, y in dataloader: ...`
        subsets.append(DataLoader(Subset(dataset, subset_indices),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  generator=g))
        start = end

    return subsets
