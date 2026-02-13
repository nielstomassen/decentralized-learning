#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import glob


def build_er_graph(n, p, seed):
    G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    # ensure all nodes exist (usually already true for nx ER, but keep safe)
    for i in range(n):
        if i not in G:
            G.add_node(i)
    return G


def build_star_graph(n):
    """
    Build a star with node 0 as center and nodes 1..n-1 as leaves.
    networkx.star_graph(k) creates nodes 0..k with 0 as center, so k = n-1.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        G = nx.Graph()
        G.add_node(0)
        return G
    return nx.star_graph(n - 1)


def build_topology(args):
    if args.topology == "er":
        if args.er_p is None:
            raise ValueError("--er_p is required when --topology er")
        return build_er_graph(args.n, args.er_p, args.seed)

    if args.topology == "star":
        return build_star_graph(args.n)

    raise NotImplementedError(f"Unsupported topology: {args.topology}")


def main(args):
    # -------- load & concatenate CSVs --------
    csv_files = []
    for pat in args.csvs:
        csv_files.extend(glob.glob(pat))
    if not csv_files:
        raise ValueError("No CSVs matched.")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["seed_file"] = Path(f).name
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # -------- select round --------
    df_r = df[df["round"] == args.round]
    if df_r.empty:
        raise ValueError(f"No data for round={args.round}")

    # -------- average metric per node --------
    if args.metric not in df_r.columns:
        raise ValueError(f"Metric '{args.metric}' not in CSV.")

    agg = (
        df_r.groupby("node_id")[args.metric]
        .mean()
        .reset_index()
    )

    node_ids = agg["node_id"].astype(int).tolist()
    values = agg[args.metric].values

    # -------- build topology --------
    G = build_topology(args)

    # -------- layout --------
    pos = nx.spring_layout(G, seed=args.layout_seed)

    # -------- labels --------
    labels = {}
    for nid, val in zip(node_ids, values):
        if args.acc_in_0_1:
            labels[nid] = f"{nid}\n{val:.2f}"
        else:
            labels[nid] = f"{nid}\n{val:.1f}%"

    # -------- plot --------
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=False, node_size=600)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    topo_line = f"{args.topology.upper()} (seed={args.seed}, {len(csv_files)} seeds)"
    if args.topology == "er":
        topo_line = f"ER (p={args.er_p}, seed={args.seed}, {len(csv_files)} seeds)"

    plt.title(
        f"Seed-averaged {args.metric} @ round {args.round}\n{topo_line}"
    )

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"[ok] Saved {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--csvs", nargs="+", required=True, help="CSV glob(s), e.g. results/*.csv")
    p.add_argument("--out", default="topology_avg.png")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--metric", default="global_test_acc")

    p.add_argument("--topology", choices=["er", "star"], required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)

    # ER-only parameter (optional at parse time, validated later)
    p.add_argument("--er_p", type=float, default=None, help="Edge prob p (required if topology=er)")

    p.add_argument("--layout_seed", type=int, default=0)
    p.add_argument("--acc_in_0_1", action="store_true")

    args = p.parse_args()
    main(args)