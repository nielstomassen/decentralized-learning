# src/mia_runner.py

from dataclasses import dataclass
from typing import Optional

import os
import torch

from mia_attacks import main_baseline
from mia_attacks import main_lira
from session_settings import MIASettings                 # your LIRA module (run_lira_attack)



class MIARunner:
    def __init__(self, config: MIASettings):
        self.config = config

    def maybe_run(
        self,
        round_nr: int,
        nodes,
        dataloaders,
        test_loader,
        model_fn,
        device: str,
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
            return None

        # Messages = what nodes send before averaging
        messages = {node.id: node.prepare_message() for node in nodes}

        if cfg.attack_type == "baseline":
            return self._run_baseline_on_message(
                round_nr,
                messages,
                dataloaders,
                test_loader,
                model_fn,
                device,
            )
        elif cfg.attack_type == "lira":
            return self._run_lira_on_message(
                round_nr,
                messages,
                dataloaders,
                test_loader,
                model_fn,
                device,
            )
        else:
            raise ValueError(f"Unknown MIA attack_type: {cfg.attack_type}")

    # ---------- concrete implementations ----------

    def _reconstruct_victim_model(self, messages, model_fn, device: str):
        victim_msg = messages[self.config.victim_id]
        victim_model = model_fn()
        victim_model.load_state_dict(victim_msg)
        victim_model.to(device)
        victim_model.eval()
        return victim_model

    def _run_baseline_on_message(
        self,
        round_nr: int,
        messages,
        dataloaders,
        test_loader,
        model_fn,
        device: str,
    ) -> dict:
        victim_model = self._reconstruct_victim_model(messages, model_fn, device)
        victim_train_loader = dataloaders[self.config.victim_id]

        results_dir = os.path.join(
            self.config.results_root,
            f"baseline_round_{round_nr + 1}_attacker_{self.config.attacker_id}_victim_{self.config.victim_id}",
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
            f"attacker {self.config.attacker_id} -> victim {self.config.victim_id}, "
            f"AUC = {result['auc']:.4f}"
        )
        return result

    def _run_lira_on_message(
        self,
        round_nr: int,
        messages,
        dataloaders,
        test_loader,
        model_fn,
        device: str,
    ) -> dict:
        victim_model = self._reconstruct_victim_model(messages, model_fn, device)
        victim_train_loader = dataloaders[self.config.victim_id]

        results_dir = os.path.join(
            self.config.results_root,
            f"lira_round_{round_nr + 1}_attacker_{self.config.attacker_id}_victim_{self.config.victim_id}",
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
            f"attacker {self.config.attacker_id} -> victim {self.config.victim_id}, "
            f"AUC = {result['auc']:.4f}"
        )
        return result
