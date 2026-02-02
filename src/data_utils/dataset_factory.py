# dataset_factory.py

from typing import Tuple, List
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .partitioner import partition_dataset  

class DatasetFactory:
    @staticmethod
    def create(
        dataset_name: str,
        num_nodes: int,
        train_batch_size: int,
        val_batch_size: int,
        partioner: str,
        alpha: float,
        no_samples: int,
        seed: int,
        data_root: str = "./data",       
    ) -> Tuple[list[DataLoader], list[DataLoader]]:

        """
        Create train dataloaders (partitioned per node) and a test dataloader.

        Args:
            dataset_name: "mnist" or "cifar10" (case-insensitive)
            num_nodes:    number of federated nodes
            train_batch_size: batch size for local training
            val_batch_size:   batch size for validation / test
            data_root:   where to store/download the data

        Returns:
            dataloaders: list of node-local DataLoaders (length = num_nodes)
            test_loader: global test DataLoader
        """
        name = dataset_name.lower()

        if name == "mnist":
            transform = transforms.Compose([
                transforms.ToTensor(),
                # standard MNIST normalization
                transforms.Normalize((0.1307,), (0.3081,)),
            ])

            train_dataset = datasets.MNIST(
                root=data_root,
                train=True,
                download=True,
                transform=transform,
            )
            test_dataset = datasets.MNIST(
                root=data_root,
                train=False,
                download=True,
                transform=transform,
            )

        elif name in ("cifar10", "cifar"):
            train_transform = transforms.Compose([
                # add noise to increase effective dataset (makes training harder but less overfitting)
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                # standard cifar10 normalization
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616],
                ),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616],
                ),
            ])

            train_dataset = datasets.CIFAR10(
                root=data_root,
                train=True,
                download=True,
                transform=train_transform,
            )
            test_dataset = datasets.CIFAR10(
                root=data_root,
                train=False,
                download=True,
                transform=test_transform,
            )

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        global_test_loader = DataLoader(
            test_dataset,
            batch_size=val_batch_size,
            shuffle=False,
        )

        # Node-local train loaders
        train_loaders, holdout_loaders = partition_dataset(
            train_dataset,
            num_nodes=num_nodes,
            batch_size=train_batch_size,
            strategy=partioner,
            dirichlet_alpha=alpha,
            dirichlet_samples_per_node=no_samples,
            seed=seed
        )

        return train_loaders, holdout_loaders, global_test_loader
