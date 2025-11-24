from .node import Node
from .utils import evaluate

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
        )
        nodes.append(node)

    return nodes

def local_training_phase(nodes, learning_settings, device: str) -> None:
    for node in nodes:
        node.local_train(
            local_epochs=learning_settings.local_epochs,
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

def run_round(round_nr: int, nodes, settings, test_loader, device: str):
    print(f"Round {round_nr + 1}")

    # 1) Local training
    local_training_phase(nodes, settings.learning, device)

    # 2) Eval before averaging (using node 0 as reference)
    do_eval = settings.enable_evaluation and ((round_nr + 1) % settings.eval_interval == 0)
    acc_before = None
    acc_after = None

    if(do_eval):
        acc_before = evaluate(nodes[0].model, test_loader, device=device)

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