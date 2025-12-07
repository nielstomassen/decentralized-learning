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

    parser.add_argument(
        "--mia-attack",
        type=str,
        default="none",
        choices=["none", "baseline", "lira"],
        help="Type of MIA attack to run each interval."
    )
    parser.add_argument("--results-root", type=str, default="results_mia", help="Location to store mia results (config + auc plots).")
    parser.add_argument("--mia-interval", type=int, default=1, help="Round interval to run mia attack.")
    parser.add_argument("--mia-attacker", type=int, default=0, help="Node id of attacker")
    parser.add_argument("--mia-victim", type=int, default=1, help="Node id of victim")
    parser.add_argument("--mia-measurement-number", type=int, default=100, help="Number of member samples used for MIA evaluation (same number of non-members is used). Total measurement set size = 2 * measurement number.")
    parser.add_argument("--mia-baseline-type", type=str, default="loss", choices=["conf", "loss", "prob"], help="Type of knowledge possessed by attacker: full training and testing losses/confidence scores/probability vector of target model.")
    parser.add_argument("--lira-known-member-perc", type= float, default=0.0, help="Percentage of victim node training data that is known as member to the attacker to train shadow model.")
    parser.add_argument("--lira-known-nonmember-perc", type=float, default=0.1, help="Non-member fraction used for shadow training. A value of 0.1 means the attacker knows non-member samples equal to 10%% of the victim's training set size (drawn from test distribution).")
    parser.add_argument("--lira-num-shadow-models", type=int, default=5, help="Amount of shadow models the attacker trains.")
    parser.add_argument("--lira-shadow-model-lr", type=float, default=1e-3, help="Learning rate used by the attacker to train shadow model.")
    parser.add_argument("--lira-shadow-model-epochs", type=int, default=5, help="Number of epochs used by the attacker when training the shadow model.")



    args = parser.parse_args()
    # if args.dataset == "mnist":
    #     args.learning_rate = 0.004
    #     args.momentum = 0

    return args