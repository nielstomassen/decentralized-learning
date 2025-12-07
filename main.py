import torch
from src.mia_runner import MIARunner
from src.topologies.topology_factory import TopologyFactory
from src.data_utils.dataset_factory import DatasetFactory
from src.models.model_factory import ModelFactory
from src.utils import set_global_seed, get_torch_device
from src.decentralized_training import create_nodes, run_round
import time
from args import get_args
from session_settings import MIASettings, SessionSettings, LearningSettings
import mia_attacks

def build_settings():
    args = get_args()
    learning_settings = LearningSettings(
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
    )
    settings = SessionSettings(
        learning=learning_settings,
        dataset=args.dataset,
        model=args.model,
        topology=args.topology,
        participants=args.peers,
        rounds=args.rounds,
        seed=args.seed,
        enable_evaluation=args.enable_evaluation,
        eval_interval=args.eval_interval,
        validation_batch_size=args.validation_batch_size,
        time_rounds=args.time_rounds,
        torch_device_name=get_torch_device(),
    )
    mia_settings = MIASettings(
        attack_type=args.mia_attack,   
        interval=args.mia_interval,    
        attacker_id=args.mia_attacker, 
        victim_id=args.mia_victim,     
        measurement_number=args.mia_measurement_number,
        results_root=args.results_root,
        mia_baseline_type=args.mia_baseline_type,
        lira_known_member_perc = args.lira_known_member_perc,
        lira_known_nonmember_perc = args.lira_known_nonmember_perc,
        lira_num_shadow_models = args.lira_num_shadow_models,
        lira_shadow_model_lr = args.lira_shadow_model_lr,
        lira_shadow_model_epochs = args.lira_shadow_model_epochs,

    )
    return settings, learning_settings, mia_settings

def run():
    settings, learning_settings, mia_settings = build_settings()
    mia_runner = MIARunner(mia_settings) if mia_settings.attack_type != "none" else None
    set_global_seed(settings.seed)

    # 1. Load dataset
    dataloaders, test_loader = DatasetFactory.create(
        dataset_name=settings.dataset,
        num_nodes=settings.participants,
        train_batch_size=learning_settings.batch_size,
        val_batch_size=settings.validation_batch_size,
        data_root="./data"
    )

    # 2. Build topology
    topology = TopologyFactory.create(
        settings.topology, 
        num_nodes=settings.participants,
        shuffle_nodes=False
    )  
    topology.draw(save_path="plots/" + settings.topology)
    
    # 3. Initialize nodes
    model_fn = ModelFactory.create(settings.model, settings.dataset)
    nodes = create_nodes(settings, dataloaders, topology, model_fn)
    print(settings.torch_device_name)
    # 4. Training loop
    for rnd in range(settings.rounds):
        if(settings.time_rounds):
            start = time.perf_counter()
        run_round(
            round_nr=rnd,
            nodes=nodes,
            settings=settings,
            test_loader=test_loader,
            device=settings.torch_device_name,
            dataloaders=dataloaders,
            model_fn=model_fn,
            mia_runner=mia_runner
        )
        if(settings.time_rounds):
            # Make sure all GPU work is finished before stopping the timer
            if "cuda" in settings.torch_device_name and torch.cuda.is_available():
                torch.cuda.synchronize() 
            end = time.perf_counter()
            duration = end - start
            print(f"The round took {duration:.3f} seconds")
 
if __name__ == "__main__":
    run()
