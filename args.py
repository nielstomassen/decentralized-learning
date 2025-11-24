import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # Learning settings
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate for the optimizer (SGD).')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum factor for SGD.')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='L2 weight decay (regularization) for SGD.')
    parser.add_argument('--batch-size', type=int, default=32, help='Local training batch size per peer.')
    parser.add_argument('--local-epochs', type=int, default=1, help='Number of local epochs per round on each peer.')

    # Important
    parser.add_argument('--dataset', type=str, default="mnist", choices=['mnist', 'cifar10'], help='Dataset to use.')
    parser.add_argument('--model', type=str, default="cnn", choices=['cnn', 'logreg'], help='Model type to use.')
    parser.add_argument('--topology', type=str, default="ring", choices=["ring", "full", "fully_connected", "clique", "er", "erdos_renyi", "small_world", "ws", "star", "hub", "grid", "mesh", "lattice"], help='Communication topology between peers.')
    parser.add_argument('--peers', type=int, default=10, help='Number of participants (nodes/clients).')
    parser.add_argument('--rounds', type=int, default=5, help='Number of training rounds.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--eval', dest='enable_evaluation', action='store_true', help='Enable evaluation during training (default: disabled).')
    parser.add_argument('--eval-interval', type=int, default=1, help='Evaluate every N rounds (only used if --eval is set).')
    parser.add_argument('--validation-batch-size', type=int, default=256, help='Batch size used during evaluation.')
    parser.add_argument('--time-rounds', dest='time_rounds', action='store_true', help='Measure and print the duration of each training round.')
    # Add later 
    # parser.add_argument('--partitioner', type=str, default="iid", choices=["iid", "shards", "dirichlet"])
    # parser.add_argument('--log-level', type=str, default="INFO") 
    # parser.add_argument('--alpha', type=float, default=0.1)
    # parser.add_argument('--algorithm', type=str, default="dpsgd", choices=["fl", "dpsgd", "gossip", "super-gossip", "adpsgd", "epidemic", "lubor", "conflux", "teleportation", "shatter"])
    # parser.add_argument('--validation-set-fraction', type=float, default=0.0)
    # parser.add_argument('--compute-validation-loss-global-model', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    # if args.dataset == "mnist":
    #     args.learning_rate = 0.004
    #     args.momentum = 0

    return args