from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json


@dataclass
class LearningSettings:
    """
    Settings related to the learning process.
    """
    learning_rate: float
    momentum: float
    weight_decay: float
    batch_size: int
    local_steps: int


@dataclass_json
@dataclass
class SessionSettings:
    """
    All settings related to a training session.
    """
    learning: LearningSettings
    dataset: str
    model: str
    topology: str
    participants: int
    rounds: int
    seed: int   
    validation_batch_size: int = 256
    torch_device_name: str = "cpu"

    # algorithm: str
    # alpha: float = 1
    # validation_set_fraction: float = 0
    # compute_validation_loss_global_model: bool = False    
    # partitioner: str = "iid"  # iid, shards or dirichlet
    # log_level: str = "INFO"