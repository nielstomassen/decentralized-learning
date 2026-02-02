import os
import torch
from src.mia_runner import MIARunner
from src.plots import draw_topology, plot_mia_curves
from src.topologies.topology_factory import TopologyFactory
from src.data_utils.dataset_factory import DatasetFactory
from src.models.model_factory import ModelFactory
from src.utils import set_global_seed, get_torch_device
from src.decentralized_training import create_nodes, run_round
import time
from args import get_args
from session_settings import MIASettings, SessionSettings, LearningSettings, TopologySettings
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
        mia_results_root = args.mia_results_root,
    )
    topology_settings = TopologySettings(
        topology_name = args.topology,
        regular_degree = args.regular_degree,
        er_p=args.er_p,
        sbm_num_blocks=args.sbm_num_blocks,
        sbm_p_in=args.sbm_p_in,
        sbm_p_out=args.sbm_p_out,
        ws_k=args.ws_k,
        ws_p=args.ws_p,
        topology_seed=args.topology_seed,
    )
    settings = SessionSettings(
        learning=learning_settings,
        mia=mia_settings,
        topology=topology_settings,
        dataset=args.dataset,
        model=args.model,
        partitioner=args.partitioner,
        alpha=args.alpha,
        beta=args.beta,
        message_type=args.message_type,
        no_samples=args.no_samples,
        participants=args.peers,
        rounds=args.rounds,
        seed=args.seed,
        enable_dp=args.enable_dp,
        dp_noise_multiplier=args.dp_noise_multiplier,
        dp_max_grad_norm=args.dp_max_grad_norm,
        dp_delta=args.dp_delta,
        enable_chunking=args.enable_chunking,
        enable_evaluation=args.enable_evaluation,
        eval_interval=args.eval_interval,
        validation_batch_size=args.validation_batch_size,
        time_rounds=args.time_rounds,
        torch_device_name=get_torch_device(),
    )
    
    return settings, learning_settings, mia_settings, topology_settings

def run():
    settings, learning_settings, mia_settings, topology_settings = build_settings()
    mia_runner = MIARunner(mia_settings) if mia_settings.attack_type != "none" else None
    set_global_seed(settings.seed)
    print(settings)
    # 1. Load dataset
    dataloaders, test_loaders, global_test_loader = DatasetFactory.create(
        dataset_name=settings.dataset,
        num_nodes=settings.participants,
        train_batch_size=learning_settings.batch_size,
        val_batch_size=settings.validation_batch_size,
        data_root="./data",
        partioner=settings.partitioner,
        alpha=settings.alpha,
        no_samples=settings.no_samples,
        seed=settings.seed
    )
    sizes = [len(dl.dataset) for dl in dataloaders]
    print("Per-node train samples:",
        "size", min(sizes),
        "sum", sum(sizes))


    # 2. Build topology
    topology = TopologyFactory.create_from_settings(
        topology_settings=topology_settings, 
        num_nodes=settings.participants,
        shuffle_nodes=False,
    )
    draw_topology(save_root="plots/topologies/", topology=topology, topology_name=topology_settings.topology_name)
    
    # 3. Initialize nodes
    model_fn = ModelFactory.create(settings.model, settings.dataset)
    nodes = create_nodes(settings, dataloaders, topology, model_fn)
    
    # 4. Training loop
    for rnd in range(settings.rounds):
        if(settings.time_rounds):
            start = time.perf_counter()
        run_round(
            round_nr=rnd,
            topology=topology,
            nodes=nodes,
            settings=settings,
            test_loaders=test_loaders,
            device=settings.torch_device_name,
            dataloaders=dataloaders,
            model_fn=model_fn,
            mia_runner=mia_runner,
            global_test_loader=global_test_loader
        )
        if(settings.time_rounds):
            # Make sure all GPU work is finished before stopping the timer
            if "cuda" in settings.torch_device_name and torch.cuda.is_available():
                torch.cuda.synchronize() 
            end = time.perf_counter()
            duration = end - start
            print(f"The round took {duration:.3f} seconds")
    
    # plot_mia_curves(mia_runner, settings)
    mia_runner.save_csv(settings=settings)

 
if __name__ == "__main__":
    run()
