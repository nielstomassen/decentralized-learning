from torch.utils.data import DataLoader, Subset
import numpy as np
import torch
import random
from collections import defaultdict

def build_classes_dict(dataset):
    """
    Returns: dict[label] -> list of indices with that label
    """
    classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        if torch.is_tensor(label):
            # robust: scalar tensor or shape [1]
            label = label.item()
        if label in classes:
            classes[label].append(ind)
        else:
            classes[label] = [ind]
    return classes

def split_train_holdout(indices_per_node, holdout_frac: float, seed: int):
    rng = np.random.RandomState(seed)
    train_indices_per_node = []
    holdout_indices_per_node = []

    for node_id, inds in enumerate(indices_per_node):
        inds = np.array(inds)
        rng.shuffle(inds)  # deterministic
        n_holdout = int(round(len(inds) * holdout_frac))
        holdout = inds[:n_holdout].tolist()
        train = inds[n_holdout:].tolist()

        train_indices_per_node.append(train)
        holdout_indices_per_node.append(holdout)

    return train_indices_per_node, holdout_indices_per_node

def make_dataloaders_from_indices(dataset, indices_per_node, batch_size: int = 32, seed: int = 123):
    """
    indices_per_node: list[list[int]] of length num_nodes
    """
    loaders = []
    for node_id, subset_indices in enumerate(indices_per_node):
        # per-node generator, but deterministic
        g = torch.Generator()
        g.manual_seed(seed + node_id)

        loaders.append(
            DataLoader(
                Subset(dataset, subset_indices),
                batch_size=batch_size,
                shuffle=True,
                generator=g,
            )
        )
    return loaders

def iid_partition_indices(dataset_len: int, num_nodes: int, shuffle: bool = True):
    # Create array [0, 1, ..., len(dataset) -1]
    indices = np.arange(dataset_len)
    if shuffle:
        # Randomly shuffle the array (seeded)
        np.random.shuffle(indices)
    # Create array of with split sizes for each node
    split_sizes = [len(indices) // num_nodes] * num_nodes
    # If not dividable by num_nodes add 1 to first n (remainder) split sizes
    # to ensure no samples are lost.
    for i in range(len(indices) % num_nodes):
        split_sizes[i] += 1

    node_indices = []
    start = 0
    for size in split_sizes:
        end = start + size
        node_indices.append(indices[start:end].tolist())
        start = end

    return node_indices

def dirichlet_partition_indices(
    dataset,
    num_nodes: int,
    no_samples: int,
    alpha: float = 0.1,
):
    data_classes = build_classes_dict(dataset)  # label -> list of indices
    # assumes all classes equal size
    class_size = len(data_classes[0])
    per_participant_list = defaultdict(list)
    per_samples_list = defaultdict(list)
    no_classes = len(data_classes.keys())

    # --- Dirichlet allocation per class ---
    for n in range(no_classes):
        image_num = []
        # Shuffle otherwise low-index nodes would always get early samples
        random.shuffle(data_classes[n])
        # How many samples each node gets of this class
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array([alpha] * num_nodes))

        for node in range(num_nodes):
            # How many samples this node gets of this class (rounded)
            no_imgs = int(round(sampled_probabilities[node]))
            # the first no_imgs remaining samples for class n
            sampled_list = data_classes[n][:min(len(data_classes[n]), no_imgs)]
            image_num.append(len(sampled_list))
            per_participant_list[node].extend(sampled_list)
            # Remove samples that were sampled from data_classes
            data_classes[n] = data_classes[n][min(len(data_classes[n]), no_imgs):]

    # IF not passed len(dataset) is used as an upperbound.
    # Will always get reduced to the smallest number of sampled images for any node
    if(no_samples is None):
        no_samples = len(dataset)
    # --- Adjust global no_samples to lowest number of samples on a node
    for i in range(num_nodes):
        no_samples = min(no_samples, len(per_participant_list[i]))

    # --- Sample exactly no_samples per client -> same dataset size for every node
    for i in range(num_nodes):
        sample_index = np.random.choice(
            len(per_participant_list[i]),
            no_samples,
            replace=False,
        )
        per_samples_list[i].extend(
            np.array(per_participant_list[i])[sample_index]
        )

    # Convert dict to list[list[int]] in user-id order
    indices_per_node = [per_samples_list[i] for i in range(num_nodes)]
    return indices_per_node

def partition_dataset(
    dataset,
    num_nodes: int,
    seed: int,
    batch_size: int = 32,
    strategy: str = "iid",                  # "iid" or "dirichlet"
    dirichlet_alpha: float = 0.1,
    dirichlet_samples_per_node: int | None = None,
    shuffle: bool = True,
):
    """
    High-level partition function.

    strategy:
        - "iid": nearly equal random split
        - "dirichlet": label-skew via Dirichlet(alpha)

    dirichlet_alpha:
        smaller -> more skew; larger -> closer to iid

    dirichlet_samples_per_node:
        max number of samples per client in Dirichlet case (None = keep all)
    """
    random.seed(seed)
    np.random.seed(seed)
    if strategy == "iid":
        indices_per_node = iid_partition_indices(len(dataset), num_nodes, shuffle=shuffle)

    elif strategy == "dirichlet":
        indices_per_node = dirichlet_partition_indices(
            dataset,
            num_nodes=num_nodes,
            no_samples=dirichlet_samples_per_node,
            alpha=dirichlet_alpha,
        )
    else:
        raise ValueError(f"Unknown partition strategy: {strategy}")

    train_idx, holdout_idx = split_train_holdout(
            indices_per_node,
            holdout_frac=0.2,   
            seed=seed + 12345   # <-- offset to avoid coupling with earlier shuffles
    )  

    train_loaders = make_dataloaders_from_indices(dataset, train_idx, batch_size=batch_size, seed=seed)
    holdout_loaders = make_dataloaders_from_indices(dataset, holdout_idx, batch_size=batch_size, seed=seed+999)

    return train_loaders, holdout_loaders

