import os
import torch
import numpy as np
import random

def set_global_seed(seed: int = 42):
    _enable_cudnn_benchmark()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def _enable_cudnn_benchmark():
    """Enable cuDNN benchmark for faster GPU training (slight non-determinism). Set env CUDA_BENCHMARK=1 to enable."""
    if torch.cuda.is_available() and os.environ.get("CUDA_BENCHMARK", "").strip() in ("1", "true", "yes"):
        torch.backends.cudnn.benchmark = True

def get_torch_device() -> str:
    # return "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def evaluate(model, dataloader, device="cpu", top_k=1):
    """Evaluate a model on a given dataloader and return accuracy in [0, 1].
    A prediction is correct if the true class is in the top-k predicted classes (default top-1)."""
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            k = min(top_k, logits.size(1))
            if k == 1:
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
            else:
                _, topk_preds = torch.topk(logits, k, dim=1)
                correct += (topk_preds == y.unsqueeze(1)).any(dim=1).sum().item()
            total += y.size(0)

    return correct / total if total > 0 else 0.0