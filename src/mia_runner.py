# src/mia_runner.py

import csv
from dataclasses import dataclass
from typing import Any, Dict, Optional

import os
import torch

from mia_attacks import main_baseline
from mia_attacks import main_lira
from session_settings import MIASettings, SessionSettings
from src.utils import set_global_seed                 # your LIRA module (run_lira_attack)



class MIARunner:
    def __init__(self, config: MIASettings):
        self.config = config

        # each entry corresponds to a round; value is dict[node_id -> metric]
        self.aucs: list[dict] = []
        self.train_accs: list[dict] = []
        self.test_accs: list[dict] = []

    def maybe_run(
        self,
        round_nr: int,
        nodes,
        dataloaders,
        test_loader,
        model_fn,
        device: str,
        seed,
        message_type,
    ) -> Optional[dict]:
        """
        Run the configured MIA attack for this round, if enabled and due.

        Returns attack result dict or None.
        """
        cfg = self.config

        if cfg.attack_type == "none":
            return None

        # interval check
        if (round_nr + 1) % cfg.interval != 0:
            self.aucs.append({})
            self.train_accs.append({})
            self.test_accs.append({})
            return None

        # Messages = what nodes send before averaging
        messages = {node.id: node.prepare_message() for node in nodes}
        round_results: Dict[Any, dict] = {}
        round_aucs: Dict[Any, float] = {}
        round_train_accs: Dict[Any, float] = {}
        round_test_accs: Dict[Any, float] = {}

        for victim_id in messages.keys():
            # Reseed for consistent mia attacks
            set_global_seed(seed)
            result = self._run_attack_for_victim(
                round_nr=round_nr,
                victim_id=victim_id,
                messages=messages,
                dataloaders=dataloaders,
                test_loader=test_loader,
                model_fn=model_fn,
                device=device,
                nodes=nodes,
                message_type=message_type,
            )

            round_results[victim_id] = result
            round_aucs[victim_id] = result.get("auc")
            round_train_accs[victim_id] = result.get("train_accuracy")
            round_test_accs[victim_id] = result.get("test_accuracy")

        self.aucs.append(round_aucs)
        self.train_accs.append(round_train_accs)
        self.test_accs.append(round_test_accs)

        return round_results
    
    def _run_attack_for_victim(
        self,
        round_nr: int,
        victim_id,
        messages,
        dataloaders,
        test_loader,
        model_fn,
        device: str,
        nodes,
        message_type,
    ) -> dict:
        cfg = self.config

        if cfg.attack_type == "baseline":
            result = self._run_baseline_on_message(
                round_nr,
                victim_id,
                messages,
                dataloaders,
                test_loader,
                model_fn,
                device,
                nodes,
                message_type,
            )
        elif cfg.attack_type == "lira":
            result = self._run_lira_on_message(
                round_nr,
                victim_id,
                messages,
                dataloaders,
                test_loader,
                model_fn,
                device,
                nodes,
                message_type,
            )
        else:
            raise ValueError(f"Unknown MIA attack_type: {cfg.attack_type}")
        return result

    # ---------- concrete implementations ----------

    def _reconstruct_victim_model(self, messages, victim_id, model_fn, device: str, nodes, message_type):
        victim_msg = messages[victim_id]
        victim_model = model_fn()

        if message_type == "full":
            full_state = victim_msg
        elif message_type == "delta":
            base = nodes[victim_id]._state_before_local
            if base is None:
                raise RuntimeError("Delta message but victim has no _state_before_local snapshot.")
            full_state = {k: base[k] + victim_msg[k] for k in base.keys()}
        else:
            raise ValueError(f"Unknown message_type: {message_type}")

        victim_model.load_state_dict(full_state)
        victim_model.to(device).eval()
        return victim_model


    def _run_baseline_on_message(
        self,
        round_nr: int,
        victim_id,
        messages,
        dataloaders,
        test_loader,
        model_fn,
        device: str,
        nodes,
        message_type
    ) -> dict:
        victim_model = self._reconstruct_victim_model(messages, victim_id, model_fn, device, nodes, message_type)
        victim_train_loader = dataloaders[victim_id]

        results_dir = os.path.join(
            self.config.results_root,
            f"baseline_round_{round_nr + 1}_attacker_{self.config.attacker_id}_victim_{victim_id}",
        )

        result = main_baseline.run_mia_attack(
            target_model=victim_model,
            train_loader=victim_train_loader,
            test_loader=test_loader,
            device=device,
            choice=self.config.mia_baseline_type,  
            measurement_number=self.config.measurement_number,
            results_dir=results_dir,
            save_artifacts=True,
        )

        print(
            f"[MIA-Baseline] Round {round_nr + 1}: "
            f"attacker {self.config.attacker_id} -> victim {victim_id}, "
            f"AUC = {result['auc']:.4f}"
        )
        return result

    def _run_lira_on_message(
        self,
        round_nr: int,
        victim_id,
        messages,
        dataloaders,
        test_loader,
        model_fn,
        device: str,
        nodes,
        message_type
    ) -> dict:
        victim_model = self._reconstruct_victim_model(messages, victim_id, model_fn, device, nodes, message_type)
        victim_train_loader = dataloaders[victim_id]

        results_dir = os.path.join(
            self.config.results_root,
            f"lira_round_{round_nr + 1}_attacker_{self.config.attacker_id}_victim_{victim_id}",
        )

        # LIRA ignores measurement_number in the same way, but you can pass it
        result = main_lira.run_lira_attack(
            target_model=victim_model,
            train_loader=victim_train_loader,
            test_loader=test_loader,
            device=device,
            # you can expose these as config if needed:
            perc=0.05,
            perc_test=0.01,
            measurement_number=self.config.measurement_number,
            num_shadow_models=5,
            lr_shadow_model=1e-3,
            epochs_shadow_model=10,
            results_dir=results_dir,
            save_artifacts=True,
            shadow_model_fn=model_fn
        )

        print(
            f"[MIA-LIRA] Round {round_nr + 1}: "
            f"attacker {self.config.attacker_id} -> victim {victim_id}, "
            f"AUC = {result['auc']:.4f}"
        )
        return result

    def save_csv(self, settings: SessionSettings):
        """
        Save recorded metrics to a CSV file.
        Columns: round, train_acc, test_acc, auc
        """
        filename = (
            f"{settings.dataset}_{settings.model}_{settings.topology}_"
            f"alpha{settings.alpha}_"
            f"beta{settings.beta}_"
            f"message{settings.message_type}_"
            f"seed{settings.seed}.csv"
        )
        filepath = os.path.join(self.config.mia_results_root, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Determine all node_ids seen across rounds (in case some rounds are empty)
        all_node_ids = set()
        for d in self.aucs:
            all_node_ids.update(d.keys())

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "node_id", "train_acc", "test_acc", "auc"])

            for r in range(1, len(self.aucs) + 1):
                aucs_r = self.aucs[r - 1]
                tr_r = self.train_accs[r - 1]
                te_r = self.test_accs[r - 1]

                for node_id in sorted(all_node_ids):
                    writer.writerow([
                        r,
                        node_id,
                        tr_r.get(node_id),
                        te_r.get(node_id),
                        aucs_r.get(node_id),
                    ])

        print(f"[MIA] Saved metrics to {filepath}")