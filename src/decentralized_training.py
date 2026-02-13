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
        if getattr(settings, "enable_dp", False):
            d = max(1, len(node.neighbors))
            d0 = getattr(settings, "dp_ref_degree", 4)  # average degree (normal noise)
            scale = (d0 / d) ** 0.5

            sigma = settings.dp_noise_multiplier # * scale
            sigma = min(sigma, getattr(settings, "dp_sigma_cap", 10.0))  
            node.enable_dp_training(
                noise_multiplier=sigma,
                max_grad_norm=settings.dp_max_grad_norm,
                delta=getattr(settings, "dp_delta", 1e-5),
            )
        nodes.append(node)

    return nodes

def local_training_phase(nodes, learning_settings, device: str) -> None:
    for node in nodes:
        node.local_train(
            local_epochs=learning_settings.local_epochs,
            device=device,
        )

def communication_phase_chunked(nodes, seed: int, round_nr: int, enable_chunking: bool):
    """
    Build chunked messages, deliver them, and return them.

    Returns:
      sent: dict[sender_id -> dict[receiver_id -> chunk_dict]]
    """
    sent = {}

    for sender in nodes:
        # Deterministic per (seed, round, sender)
        s = seed * 10_000_000 + round_nr * 10_000 + sender.id
        sent[sender.id] = sender.prepare_messages_for_neighbors_rowblocks(seed=s, enable_chunking=enable_chunking)

    # Deliver exactly what was sent
    for receiver in nodes:
        for nb in receiver.neighbors:  # nb is sender id
            chunk = sent[nb].get(receiver.id, {})
            receiver.receive_message(nb, chunk)

    return sent


def averaging_phase(nodes):
    for node in nodes:
        node.average_with_neighbors()

def run_round(round_nr: int, topology, nodes, settings, test_loaders, device: str, dataloaders, model_fn, global_test_loader, mia_runner=None):
    print(f"Round {round_nr + 1}")
    
    # Save state before training
    pre_comm_states = {n.id: {k: v.detach().clone() for k,v in n.model.state_dict().items()} for n in nodes}
    # 1) Local training
    local_training_phase(nodes, settings.learning, device)

    do_eval = settings.enable_evaluation and ((round_nr + 1) % settings.eval_interval == 0)
    acc_before = acc_after = None
    if do_eval:
        acc_before = evaluate(nodes[0].model, global_test_loader, device=device)

    # Save training RNG state 
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()

    
    # 2) Build + deliver chunked messages ONCE
    sent = communication_phase_chunked(nodes, seed=settings.seed, round_nr=round_nr, enable_chunking=settings.enable_chunking)

    # 3) Run MIA using the SAME sent messages (attacker-view)
    if mia_runner is not None:
        # Choose ONE attacker neighbor per victim (threat model)
        attackers_for_victim = {}
        for victim in nodes:
            if not victim.neighbors:
                continue
            if getattr(settings.mia, "one_attacker", False):
                # deterministic choice (good for reproducibility)
                rng = random.Random(settings.seed + round_nr + victim.id)
                attackers_for_victim[victim.id] = [rng.choice(list(victim.neighbors))]
            else:
                attackers_for_victim[victim.id] = list(victim.neighbors)
            


        mia_runner.maybe_run(
            round_nr=round_nr,
            topology=topology,
            pre_comm_states=pre_comm_states,
            nodes=nodes,
            dataloaders=dataloaders,
            test_loaders=test_loaders,
            model_fn=model_fn,
            device=device,
            seed=settings.seed,
            message_type=settings.message_type,
            sent_messages=sent,
            attackers_for_victim=attackers_for_victim,
            global_test_loader=global_test_loader,
        )

    # Restore training RNG state
    torch.set_rng_state(torch_state)
    np.random.set_state(np_state)
    random.setstate(py_state)

    # 4) Averaging 
    averaging_phase(nodes)

    # 5) Eval after averaging
    if do_eval:
        acc_after = evaluate(nodes[0].model, global_test_loader, device=device)
        print("before: " + str(acc_before))
        print("after: " + str(acc_after))
        print("----")
        eps0 = nodes[0].get_epsilon()
        print("node0 epsilon:", eps0)

    return acc_before, acc_after