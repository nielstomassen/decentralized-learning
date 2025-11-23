import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # Learning settings
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--local-steps', type=int, default=10)

    # Important
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--model', type=str, default="cnn")
    parser.add_argument('--topology', type=str, default="ring", choices=["ring", "full", "fully_connected", "clique", "er", "erdos_renyi", "small_world", "ws"])
    parser.add_argument('--peers', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-batch-size', type=int, default=256)
    
    # Add later 
    parser.add_argument('--partitioner', type=str, default="iid", choices=["iid", "shards", "dirichlet"])
    parser.add_argument('--log-level', type=str, default="INFO") 
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--algorithm', type=str, default="dpsgd", choices=["fl", "dpsgd", "gossip", "super-gossip", "adpsgd", "epidemic", "lubor", "conflux", "teleportation", "shatter"])
    parser.add_argument('--validation-set-fraction', type=float, default=0.0)
    parser.add_argument('--compute-validation-loss-global-model', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    # if args.dataset == "mnist":
    #     args.learning_rate = 0.004
    #     args.momentum = 0

    return args