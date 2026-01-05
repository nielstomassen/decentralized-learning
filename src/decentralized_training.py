import random
import numpy as np
import torch
from .node import Node
from .utils import evaluate, set_global_seed
from mia_attacks import main_baseline
import os

def create_nodes(settings, dataloaders, topology, model_fn):
    # shared init
    global_init_model = model_fn()
    shared_init = global_init_model.state_dict()

    nodes = []
    for i in range(settings.participants):
        node = Node(
            node_id=i,
            model_fn=model_fn,
            dataloader=dataloaders[i],
            neighbors=topology.neighbors[i],
            settings=settings,
            global_init=shared_init,
            neighbor_weights=topology.weights[i],
        )
        nodes.append(node)

    return nodes

def local_training_phase(nodes, learning_settings, device: str) -> None:
    for node in nodes:
        node.local_train(
            local_epochs=learning_settings.local_epochs,
            device=device,
        )

def communication_phase(nodes, messages=None):
    """
    Prepare and deliver messages between nodes.

    If `messages` is None, we call `prepare_message()` on each node.
    """
    if messages is None:
        messages = {node.id: node.prepare_message() for node in nodes}

    for node in nodes:
        for nb in node.neighbors:
            node.receive_message(nb, messages[nb])

    return messages

def averaging_phase(nodes):
    for node in nodes:
        node.average_with_neighbors()

def run_round(round_nr: int, nodes, settings, test_loader, device: str, dataloaders, model_fn, mia_runner=None,):
    print(f"Round {round_nr + 1}")

    # 1) Local training
    local_training_phase(nodes, settings.learning, device)

    # 2) Eval before averaging (using node 0 as reference)
    do_eval = settings.enable_evaluation and ((round_nr + 1) % settings.eval_interval == 0)
    acc_before = None
    acc_after = None
    print(sum(p.abs().sum().item() for p in nodes[0].model.parameters()))
    if(do_eval):
        acc_before = evaluate(nodes[0].model, test_loader, device=device)

    # 2) (Optional) run MIA before communication/averaging, but only every k rounds
    
    # 3) Optional: run MIA on messages before communication/averaging
    # Save training rng state
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()
    
    if mia_runner is not None:
        mia_runner.maybe_run(
            round_nr=round_nr,
            nodes=nodes,
            dataloaders=dataloaders,
            test_loader=test_loader,
            model_fn=model_fn,
            device=device,
            seed=settings.seed,
            message_type=settings.message_type,
        )
    # Reload training rng state
    torch.set_rng_state(torch_state)
    np.random.set_state(np_state)
    random.setstate(py_state)
    # 3) Communication
    communication_phase(nodes)

    # 4) Averaging
    averaging_phase(nodes)

    # 5) Eval after averaging
    if(do_eval):
        acc_after = evaluate(nodes[0].model, test_loader, device=device)
        print(
            f"  Node 0 test accuracy: "
            f"before avg = {acc_before*100:.2f}%, after avg = {acc_after*100:.2f}%"
        )

    return acc_before, acc_after