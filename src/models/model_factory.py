# models_factory.py

from .mnist_cnn import MnistCNN
from .cifar10_cnn import Cifar10CNN

class ModelFactory:
    @staticmethod
    def create(model_name: str, dataset_name: str):
        """
        Return a callable model_fn() that builds a fresh model instance.

        Args:
            model_name: architecture type, e.g. "cnn", "resnet", "logreg"
            dataset_name: dataset name, e.g. "mnist", "cifar10"

        Example:
            model_fn = ModelFactory.create(model_name="cnn", dataset_name="mnist")
            model = model_fn()
        """
        mt = model_name.lower()
        ds = dataset_name.lower()

        # --- CNNs ---
        if mt in ("cnn", "conv", "convnet"):
            if ds == "mnist":
                def model_fn():
                    return MnistCNN(in_channels=1, num_classes=10)
                return model_fn

            elif ds in ("cifar10", "cifar"):
                def model_fn():
                    return Cifar10CNN(in_channels=3, num_classes=10)
                return model_fn

            else:
                raise ValueError(f"CNN not supported for dataset: {dataset_name}")

        # --- ResNet (to be implemented later) ---
        if mt in ("resnet", "resnet18", "resnet20"):
            raise ValueError(
                f"ResNet models not implemented yet for dataset '{dataset_name}'. "
                "Add the appropriate ResNet class and wiring in ModelFactory.create()."
            )

        # --- Logistic Regression (to be implemented later) ---
        if mt in ("logreg", "logistic", "logistic_regression"):
            raise ValueError(
                f"Logistic regression models not implemented yet for dataset '{dataset_name}'. "
                "Add the appropriate classifier and wiring in ModelFactory.create()."
            )

        # Unknown model type
        raise ValueError(f"Unknown model type: {model_name}")
