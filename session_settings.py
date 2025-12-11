from dataclasses import asdict, dataclass
from dataclasses_json import dataclass_json
import json

@dataclass
class LearningSettings:
    """
    Settings related to the learning process.
    """
    learning_rate: float
    momentum: float
    weight_decay: float
    batch_size: int
    local_epochs: int

    def __str__(self):
        return json.dumps(asdict(self), indent=4)

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
    partitioner: str
    alpha: float
    no_samples: int
    participants: int
    rounds: int
    seed: int
    enable_evaluation: bool 
    eval_interval: int  
    validation_batch_size: int
    time_rounds: bool
    torch_device_name: str 

    def __str__(self):
        return json.dumps(asdict(self), indent=4)
    # algorithm: str
    # alpha: float = 1
    # validation_set_fraction: float = 0
    # compute_validation_loss_global_model: bool = False    
    # partitioner: str = "iid"  # iid, shards or dirichlet
    # log_level: str = "INFO"

@dataclass
class MIASettings:
    attack_type: str = "none"    # "none" | "baseline" | "lira"
    interval: int = 1            # run every k rounds
    attacker_id: int = 0
    victim_id: int = 1
    measurement_number: int = 100
    results_root: str = "results_mia"
    mia_baseline_type: str = "loss"
    lira_known_member_perc: float = 0.05
    lira_known_nonmember_perc: float = 0.1
    lira_num_shadow_models: int = 5
    lira_shadow_model_lr: float = 1e-3
    lira_shadow_model_epochs: int = 5

    def __str__(self):
        return json.dumps(asdict(self), indent=4)
