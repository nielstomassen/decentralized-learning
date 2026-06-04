"""Display names for ChunkDP plots (user-facing labels only). 
Used in hybrid ablation section in the thesis. 
Called by hybrid_privacy_tradeoff.py so no need to run this script separately."""

from __future__ import annotations

# Internal condition string written to summary CSVs by plotting scripts.
CHUNKDP_CONDITION = "ChunkDP"

# Legacy aliases still accepted when reading older summary CSVs.
_CHUNKDP_ALIASES = frozenset(
    {
        CHUNKDP_CONDITION,
        "DP + chunk (hybrid)",
        "Hybrid",
        "DP + chunk (Hybrid)",
        "hybrid",
        "DP + chunk",
    }
)


def is_chunkdp_condition(label: str) -> bool:
    return str(label).strip() in _CHUNKDP_ALIASES


def normalize_condition_label(label: str) -> str:
    """Map legacy hybrid names to ChunkDP for consistent tables and comparisons."""
    if is_chunkdp_condition(label):
        return CHUNKDP_CONDITION
    return str(label)


def chunkdp_xtick_label(condition: str) -> str:
    """Display labels for bar/scatter ticks (single line unless legacy alias)."""
    c = normalize_condition_label(condition)
    if c == CHUNKDP_CONDITION:
        return "ChunkDP"
    if c == "Chunk only":
        return "Topology-aware chunk only"
    if c.startswith("Fixed-K chunking (K=") or c.startswith("Standard chunking (K="):
        return c.replace("Standard chunking (K=", "Fixed-K chunking (K=", 1)
    return c


def sweep_config_legend_label(dp_noise: float, dp_max_grad_norm: float) -> str:
    return rf"ChunkDP ($\sigma$={dp_noise}, $C$={dp_max_grad_norm})"


def mean_accuracy_axis_label() -> str:
    return "Mean Top-5 Test Accuracy"


def mean_mia_auc_axis_label(auc_col: str = "max_auc") -> str:
    if auc_col == "max_auc":
        return "Mean Maximum MIA AUC"
    if auc_col == "avg_auc":
        return "Mean Average MIA AUC"
    return f"Mean MIA {auc_col}"


def node_mia_auc_axis_label(auc_col: str = "max_auc") -> str:
    if auc_col == "max_auc":
        return "Maximum MIA AUC (node-level)"
    if auc_col == "avg_auc":
        return "Average MIA AUC (node-level)"
    return f"MIA {auc_col} (node-level)"


def node_accuracy_axis_label() -> str:
    return "Top-5 test accuracy (node-level)"


def deployment_score_ylabel() -> str:
    return r"$S_\lambda = (1-\lambda)u - \lambda r$"


def deployment_score_xlabel() -> str:
    return deployment_score_ylabel()


def auc_metric_title_suffix(auc_col: str = "max_auc") -> str:
    if auc_col == "max_auc":
        return "Mean Maximum MIA AUC"
    if auc_col == "avg_auc":
        return "Mean Average MIA AUC"
    return f"mean {auc_col}"
