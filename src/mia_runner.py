# src/mia_runner.py

import csv
import os
import random

import numpy as np
import torch
import networkx as nx

from mia_attacks import main_baseline, main_lira
from session_settings import MIASettings, SessionSettings
from src.utils import set_global_seed, evaluate


class MIARunner:
    def __init__(self, config: MIASettings):
        self.config = config

        self.global_test_accs: list[dict] = []
        self.epsilons: list[dict] = []

        # Static graph properties
        self.victim_degrees: dict[int, int] = {}
        self.victim_betweenness: dict[int, float] = {}
        self.victim_closeness: dict[int, float] = {}
        self.victim_kcore: dict[int, int] = {}

        # Aggregated MIA metrics (per round, per victim)
        self.avg_aucs: list[dict] = []
        self.max_aucs: list[dict] = []
        self.argmax_attackers: list[dict] = []

        # Aggregated chunk visibility metrics
        self.avg_chunk_frac: list[dict] = []
        self.max_chunk_frac: list[dict] = []
        self.argmax_chunk_attacker: list[dict] = []

    def maybe_run(
        self,
        round_nr: int,
        topology,
        pre_comm_states,
        nodes,
        dataloaders,
        test_loaders,
        global_test_loader,
        model_fn,
        device: str,
        seed,
        message_type,
        sent_messages: dict,
        attackers_for_victim: dict,  # victim_id -> list[attacker_id]
    ):
        cfg = self.config
        if cfg.attack_type == "none":
            return None

        # Skip this round (but keep arrays aligned)
        if (round_nr + 1) % cfg.interval != 0:
            self.global_test_accs.append({})
            self.epsilons.append({})
            self.avg_aucs.append({})
            self.max_aucs.append({})
            self.argmax_attackers.append({})
            self.avg_chunk_frac.append({})
            self.max_chunk_frac.append({})
            self.argmax_chunk_attacker.append({})
            return None

        G = topology.graph
        bg = G.to_undirected() if G.is_directed() else G

        # Cache degrees once
        if not self.victim_degrees:
            self.victim_degrees = {n.id: len(n.neighbors) for n in nodes}

       # Cache centralities once
        if not self.victim_betweenness:
            bc = nx.betweenness_centrality(bg, normalized=True)
            self.victim_betweenness = {int(k): float(v) for k, v in bc.items()}

        if not self.victim_closeness:
            cc = nx.closeness_centrality(bg)
            self.victim_closeness = {int(k): float(v) for k, v in cc.items()}

        if not self.victim_kcore:
            core = nx.core_number(bg)
            self.victim_kcore = {int(k): int(v) for k, v in core.items()}

        round_eps = {}

        round_global_test_accs = {}
        # round-level aggregates per victim
        round_avg_auc = {}
        round_max_auc = {}
        round_argmax_attacker = {}
        round_avg_chunk_frac = {}
        round_max_chunk_frac = {}
        round_argmax_chunk_attacker = {}

        # store per-(victim,attacker) AUCs for debugging
        round_auc_by_attacker = {}  # victim_id -> {attacker_id: auc}
        
        for node in nodes:
            round_global_test_accs[node.id] = float(evaluate(node.model, global_test_loader, device=device))

        torch_state = torch.get_rng_state()
        np_state = np.random.get_state()
        py_state = random.getstate()

        for victim in nodes:
            victim_id = victim.id
            attacker_ids = attackers_for_victim.get(victim_id, [])
            if not attacker_ids:
                continue

            # victim-level lists to aggregate
            aucs_this_victim = []
            per_attacker_auc = {}

            eps = victim.get_epsilon() # None if no DP enabled
            round_eps[victim_id] = eps
            
            # Total elements victim sent this round (sum over all receiver chunks)
            victim_total_numel = int(
                sum(
                    part["v"].numel()
                    for payload in sent_messages[victim_id].values()   # each receiver payload
                    for part in payload.get("parts", [])
                )
            )
            chunk_frac_per_att = {}

            # Run attack for each attacker neighbor
            for attacker_id in attacker_ids:
                # EXACT chunk attacker received
                victim_chunk_seen = sent_messages[victim_id].get(attacker_id, {})
                parts = victim_chunk_seen.get("parts", [])
                if len(parts) == 0:
                    continue
                cnum = int(sum(part["v"].numel() for part in parts))
                cfrac = (float(cnum) / float(victim_total_numel)) if victim_total_numel else 0.0
                chunk_frac_per_att[attacker_id] = cfrac

                # Restore seed so every attacker has same attack behaviour
                torch.set_rng_state(torch_state)
                np.random.set_state(np_state)
                random.setstate(py_state)
                result = self._run_attack_for_victim(
                    round_nr=round_nr,
                    victim_id=victim_id,
                    attacker_id=attacker_id,
                    pre_comm_states=pre_comm_states,
                    victim_chunk_seen=victim_chunk_seen,
                    dataloaders=dataloaders,
                    test_loaders=test_loaders,
                    model_fn=model_fn,
                    device=device,
                    nodes=nodes,
                    message_type=message_type,
                )
                auc = result.get("auc")
                if auc is None:
                    continue

                aucs_this_victim.append(float(auc))
                per_attacker_auc[attacker_id] = float(auc)

            if aucs_this_victim:
                avg_auc = float(sum(aucs_this_victim) / len(aucs_this_victim))
                max_auc = float(max(aucs_this_victim))
                argmax_att = max(per_attacker_auc, key=per_attacker_auc.get)

                round_avg_auc[victim_id] = avg_auc
                round_max_auc[victim_id] = max_auc
                round_argmax_attacker[victim_id] = int(argmax_att)
                round_auc_by_attacker[victim_id] = per_attacker_auc
            else:
                round_avg_auc[victim_id] = None
                round_max_auc[victim_id] = None
                round_argmax_attacker[victim_id] = None
                round_auc_by_attacker[victim_id] = {}

            # Aggregate chunk visibility across attackers
            if chunk_frac_per_att:
                fracs = list(chunk_frac_per_att.values())
                round_avg_chunk_frac[victim_id] = float(sum(fracs) / len(fracs))

                max_att = max(chunk_frac_per_att, key=chunk_frac_per_att.get)
                round_max_chunk_frac[victim_id] = float(chunk_frac_per_att[max_att])
                round_argmax_chunk_attacker[victim_id] = int(max_att)
            else:
                round_avg_chunk_frac[victim_id] = None
                round_max_chunk_frac[victim_id] = None
                round_argmax_chunk_attacker[victim_id] = None

        self.global_test_accs.append(round_global_test_accs)
        self.epsilons.append(round_eps)
        self.avg_aucs.append(round_avg_auc)
        self.max_aucs.append(round_max_auc)
        self.argmax_attackers.append(round_argmax_attacker)
        self.avg_chunk_frac.append(round_avg_chunk_frac)
        self.max_chunk_frac.append(round_max_chunk_frac)
        self.argmax_chunk_attacker.append(round_argmax_chunk_attacker)

        return {
            "avg_auc": round_avg_auc,
            "max_auc": round_max_auc,
            "argmax_attacker": round_argmax_attacker,
        }

    def _run_attack_for_victim(
        self,
        round_nr: int,
        victim_id,
        attacker_id,
        pre_comm_states,
        victim_chunk_seen,
        dataloaders,
        test_loaders,
        model_fn,
        device: str,
        nodes,
        message_type,
    ) -> dict:
        cfg = self.config

        if cfg.attack_type == "baseline":
            return self._run_baseline_on_victim(
                round_nr,
                victim_id,
                attacker_id,
                pre_comm_states,
                victim_chunk_seen,
                dataloaders,
                test_loaders,
                model_fn,
                device,
                nodes,
                message_type,
            )

        if cfg.attack_type == "lira":
            return self._run_lira_on_victim(
                round_nr,
                victim_id,
                attacker_id,
                pre_comm_states,
                victim_chunk_seen,
                dataloaders,
                test_loaders,
                model_fn,
                device,
                nodes,
                message_type,
            )

        raise ValueError(f"Unknown MIA attack_type: {cfg.attack_type}")

    def _reconstruct_proxy_model_from_attacker_view(self, victim_id, attacker_id, pre_comm_states, victim_chunk_seen, model_fn, device, nodes, message_type):
        # base_state = {k: v.detach().clone() for k, v in nodes[attacker_id].model.state_dict().items()}
        base_state = {k: v.detach().clone() for k,v in pre_comm_states[attacker_id].items()}

        for part in victim_chunk_seen.get("parts", []):
            k = part["k"]
            v = part["v"]
            s, e = part["start"], part["end"]

            if message_type == "full":
                # Full tensor
                if s is None or e is None:
                    base_state[k] = v.detach().clone()
                # Chunked tensor
                else:
                    base_state[k][s:e] = v.detach().clone()

            elif message_type == "delta":
                if s is None or e is None:
                    base_state[k] = base_state[k] + v.detach().clone()
                else:
                    base_state[k][s:e] = base_state[k][s:e] + v.detach().clone()
            else:
                raise ValueError(f"Unknown message_type: {message_type}")

        proxy = model_fn()
        proxy.load_state_dict(base_state)
        proxy.to(device).eval()
        return proxy


    def _run_baseline_on_victim(
        self,
        round_nr: int,
        victim_id: int,
        attacker_id: int,
        pre_comm_states,
        victim_chunk_seen: dict,
        dataloaders,
        test_loaders,
        model_fn,
        device: str,
        nodes,
        message_type: str,
    ) -> dict:
        proxy_model = self._reconstruct_proxy_model_from_attacker_view(
            attacker_id=attacker_id,
            victim_id=victim_id,
            pre_comm_states=pre_comm_states,
            victim_chunk_seen=victim_chunk_seen,
            model_fn=model_fn,
            device=device,
            nodes=nodes,
            message_type=message_type,
        )

        victim_train_loader = dataloaders[victim_id]

        results_dir = os.path.join(
            self.config.results_root,
            f"baseline_round_{round_nr + 1}_attacker_{attacker_id}_victim_{victim_id}",
        )

        result = main_baseline.run_mia_attack(
            target_model=proxy_model,
            train_loader=victim_train_loader,
            test_loader=test_loaders[victim_id],
            victim_id=victim_id,
            device=device,
            choice=self.config.mia_baseline_type,
            measurement_number=self.config.measurement_number,
            results_dir=results_dir,
            save_artifacts=True,
        )

        print(
            f"[MIA-Baseline] Round {round_nr + 1}: "
            f"attacker {attacker_id} -> victim {victim_id}, AUC={result['auc']:.4f}"
        )
        return result

    def _run_lira_on_victim(
        self,
        round_nr: int,
        victim_id: int,
        attacker_id: int,
        pre_comm_states,
        victim_chunk_seen: dict,
        dataloaders,
        test_loaders,
        model_fn,
        device: str,
        nodes,
        message_type: str,
    ) -> dict:
        proxy_model = self._reconstruct_proxy_model_from_attacker_view(
            attacker_id=attacker_id,
            victim_id=victim_id,
            pre_comm_states=pre_comm_states,
            victim_chunk_seen=victim_chunk_seen,
            model_fn=model_fn,
            device=device,
            nodes=nodes,
            message_type=message_type,
        )

        victim_train_loader = dataloaders[victim_id]

        results_dir = os.path.join(
            self.config.results_root,
            f"lira_round_{round_nr + 1}_attacker_{attacker_id}_victim_{victim_id}",
        )

        result = main_lira.run_lira_attack(
            target_model=proxy_model,
            train_loader=victim_train_loader,
            test_loader=test_loaders[victim_id],
            device=device,
            perc=0.05,
            perc_test=0.01,
            measurement_number=self.config.measurement_number,
            num_shadow_models=5,
            lr_shadow_model=1e-3,
            epochs_shadow_model=10,
            results_dir=results_dir,
            save_artifacts=True,
            shadow_model_fn=model_fn,
        )

        print(
            f"[MIA-LIRA] Round {round_nr + 1}: "
            f"attacker {attacker_id} -> victim {victim_id}, AUC={result['auc']:.4f}"
        )
        return result

    def save_csv(self, settings: SessionSettings):
        """
        Save recorded metrics to a CSV file.
        """
        filename = (
            f"{settings.dataset}_{settings.model}_{settings.topology.topology_name}_{settings.topology.er_p}_"
            f"alpha{settings.alpha}_"
            f"beta{settings.beta}_"
            f"message{settings.message_type}_"
            f"seed{settings.seed}.csv"
        )
        filepath = os.path.join(self.config.mia_results_root, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Determine all node_ids seen across rounds (in case some rounds are empty)
        all_node_ids = set()
        for d in self.global_test_accs:
            all_node_ids.update(d.keys())
        for d in self.avg_aucs:
            all_node_ids.update(d.keys())

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "round",
                    "node_id",
                    "degree",
                    "betweenness",
                    "closeness",
                    "k_core",
                    "epsilon",
                    "global_test_acc",
                    "avg_auc",
                    "max_auc",
                    "argmax_attacker",
                    "avg_chunk_frac",
                    "max_chunk_frac",
                    "argmax_chunk_attacker",
                ]
            )

            num_rounds = max(
                len(self.global_test_accs),
                len(self.epsilons),
                len(self.avg_aucs),
                len(self.max_aucs),
                len(self.argmax_attackers),
                len(self.avg_chunk_frac),
                len(self.max_chunk_frac),
                len(self.argmax_chunk_attacker),
            )

            for r in range(1, num_rounds + 1):
                glob_r = self.global_test_accs[r - 1] if (r - 1) < len(self.global_test_accs) else {}
                eps_r = self.epsilons[r - 1] if (r - 1) < len(self.epsilons) else {}

                avg_auc_r = self.avg_aucs[r - 1] if (r - 1) < len(self.avg_aucs) else {}
                max_auc_r = self.max_aucs[r - 1] if (r - 1) < len(self.max_aucs) else {}
                argmax_att_r = self.argmax_attackers[r - 1] if (r - 1) < len(self.argmax_attackers) else {}

                avg_cf_r = self.avg_chunk_frac[r - 1] if (r - 1) < len(self.avg_chunk_frac) else {}
                max_cf_r = self.max_chunk_frac[r - 1] if (r - 1) < len(self.max_chunk_frac) else {}
                argmax_cf_att_r = (
                    self.argmax_chunk_attacker[r - 1] if (r - 1) < len(self.argmax_chunk_attacker) else {}
                )

                for node_id in sorted(all_node_ids):
                    deg = self.victim_degrees.get(node_id)
                    btw = self.victim_betweenness.get(node_id)
                    clo = self.victim_closeness.get(node_id)
                    kcore = self.victim_kcore.get(node_id)

                    writer.writerow(
                        [
                            r,
                            node_id,
                            deg,
                            btw,
                            clo,
                            kcore,
                            eps_r.get(node_id),
                            glob_r.get(node_id),
                            avg_auc_r.get(node_id),
                            max_auc_r.get(node_id),
                            argmax_att_r.get(node_id),
                            avg_cf_r.get(node_id),
                            max_cf_r.get(node_id),
                            argmax_cf_att_r.get(node_id),
                        ]
                    )

        print(f"[MIA] Saved metrics to {filepath}")

