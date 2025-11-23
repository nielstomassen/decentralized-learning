from .node import Node
from .utils import evaluate

def create_nodes(settings, learning_settings, dataloaders, topology, model_fn):
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
            learning_settings=learning_settings,
            global_init=shared_init,
        )
        nodes.append(node)

    # Debug: check parameter initialization consistency
    for i, n in enumerate(nodes[:3]):
        w = next(iter(n.model.state_dict().values()))
        print(f"Node {i} first weight mean:", w.mean().item())

    return nodes

def local_training_phase(nodes, learning_settings, device: str) -> None:
    for node in nodes:
        node.local_train(
            num_steps=learning_settings.local_steps,
            device=device,
        )

def communication_phase(nodes):
    """Prepare and deliver messages between nodes."""
    messages = {node.id: node.prepare_message() for node in nodes}

    for node in nodes:
        for nb in node.neighbors:
            node.receive_message(messages[nb])

def averaging_phase(nodes):
    for node in nodes:
        node.average_with_neighbors()

def run_round(round_nr: int, nodes, learning_settings, test_loader, device: str):
    print(f"Round {round_nr}")

    # 1) Local training
    local_training_phase(nodes, learning_settings, device)

    # 2) Eval before averaging (using node 0 as reference)
    acc_before = evaluate(nodes[0].model, test_loader, device=device)

    # 3) Communication
    communication_phase(nodes)

    # 4) Averaging
    averaging_phase(nodes)

    # 5) Eval after averaging
    acc_after = evaluate(nodes[0].model, test_loader, device=device)

    print(
        f"  Node 0 test accuracy: "
        f"before avg = {acc_before*100:.2f}%, after avg = {acc_after*100:.2f}%"
    )

    return acc_before, acc_after