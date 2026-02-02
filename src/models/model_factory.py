# models_factory.py

from .mnist_logreg import MnistLogReg
from .mnist_cnn import MnistCNN
from .cifar10_cnn import Cifar10CNN
from .cifar10_logreg import Cifar10LogReg
from .cifar10_resnet import resnet32_cifar_gn

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

        # --- ResNet ---
        if mt in ("resnet", "resnet32"):
            if ds in ("cifar10", "cifar"):
                def model_fn():
                    return resnet32_cifar_gn(num_classes=10, gn_groups=8)
                return model_fn


        # --- Logistic Regression  ---
        if mt in ("logreg", "logistic", "logistic_regression"):
            if ds == "mnist":
                def model_fn():
                    return MnistLogReg(in_channels=1, num_classes=10)
                return model_fn
            
            elif ds in ("cifar10", "cifar"):
                def model_fn():
                    return Cifar10LogReg(in_channels=3, num_classes=10)
                return model_fn
            
            else:
                raise ValueError(f"Logistic regression not implemented for dataset '{dataset_name}'")

        # Unknown model type
        raise ValueError(f"Unknown model type: {model_name}")
