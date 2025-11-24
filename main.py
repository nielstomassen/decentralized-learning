import torch
from src.topologies.topology_factory import TopologyFactory
from src.data_utils.dataset_factory import DatasetFactory
from src.models.model_factory import ModelFactory
from src.utils import set_global_seed, get_torch_device
from src.decentralized_training import create_nodes, run_round
import time
from args import get_args
from session_settings import SessionSettings, LearningSettings

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
    return settings, learning_settings

def run():
    settings, learning_settings = build_settings()
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
