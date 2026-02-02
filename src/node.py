import copy
import random
import torch
import math
from opacus import PrivacyEngine



class Node:
    """
    Node with:
      - local SGD training
      - chunked communication using BALANCED per-tensor row-block assignment
      - averaging that updates only the entries actually received
    """

    def __init__(self, node_id, model_fn, dataloader, neighbors, settings, global_init, neighbor_weights):
        self.id = node_id
        self.model = model_fn().to(settings.torch_device_name)
        self.dataloader = dataloader

        self.neighbors = list(neighbors)  # list[int]
        self.neighbor_weights = dict(neighbor_weights)  # dict[int -> float]

        self.beta = float(settings.beta)
        self.message_type = settings.message_type  # "full" or "delta"
        self._state_before_local = None

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=settings.learning.learning_rate,
            momentum=settings.learning.momentum,
            weight_decay=settings.learning.weight_decay,
        )

        self.privacy_engine = None
        self.dp_delta = getattr(settings, "dp_delta", 1e-5)

        self._buffer_neighbor_models: dict[int, dict] = {}
        self.model.load_state_dict(global_init)

    def enable_dp_training(self, noise_multiplier: float, max_grad_norm: float, delta: float = 1e-5):
        pe = PrivacyEngine(secure_mode=False)
        self.model, self.optimizer, self.dataloader = pe.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=float(noise_multiplier),
            max_grad_norm=float(max_grad_norm),
            poisson_sampling=False, # For faster speed, turn to True for final experiment
        )
        self.privacy_engine = pe
        self.dp_delta = float(delta)

    def get_epsilon(self):
        if self.privacy_engine is None:
            return None
        return float(self.privacy_engine.get_epsilon(delta=self.dp_delta))

    def local_train(self, local_epochs=1, device="cpu"): 
        if self.message_type == "delta":
            sd = self.model.state_dict()
            # Save stated before training to compute delta: state after training - state before training
            self._state_before_local = {k: v.detach().clone() for k, v in sd.items()}
        else:
            self._state_before_local = None

        self.model.train()

        data_iter = iter(self.dataloader)
        num_batches = len(self.dataloader)
        total_steps = local_epochs * num_batches

        for _ in range(total_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                # Next pass over the data
                data_iter = iter(self.dataloader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            self.optimizer.step()

    def receive_message(self, neighbor_id: int, neighbor_payload: dict):
        """
        neighbor_payload expected:
          {"parts": [{"k": str, "start": int|None, "end": int|None, "v": Tensor}, ...]}
        """
        self._buffer_neighbor_models[int(neighbor_id)] = neighbor_payload

    def _get_full_or_delta_state(self) -> dict[str, torch.Tensor]:
        """
        Returns a dict[str, Tensor] representing either:
          - full state_dict (message_type="full"), or
          - delta state_dict (message_type="delta")
        """
        cur = self.model.state_dict()

        if self.message_type == "full":
            return {k: v.detach().clone() for k, v in cur.items()}

        if self.message_type == "delta":
            if self._state_before_local is None:
                raise RuntimeError("Delta message requested but no pre-local snapshot found.")
            # State after training - state before training
            return {k: (cur[k] - self._state_before_local[k]).detach().clone() for k in cur.keys()}

        raise ValueError(f"Unknown message_type: {self.message_type}")

    @staticmethod
    def _split_rows(t: torch.Tensor, d: int) -> list[tuple[int, int, torch.Tensor]]:
        """
        Split tensor along dim=0 into d nearly-equal contiguous row blocks.
        Returns list of (start, end, slice_tensor). Always returns exactly d blocks.
        """
        n0 = t.size(0)
        base = n0 // d
        rem = n0 % d
        blocks = []
        s = 0
        for i in range(d):
            e = s + base + (1 if i < rem else 0)
            blocks.append((s, e, t[s:e].detach().clone()))
            s = e
        return blocks

    def prepare_messages_for_neighbors_rowblocks(
        self,
        seed: int | None = None,
        split_threshold_numel: int = 1,
        broadcast_unsplittable: bool = False,
        enable_chunking: bool = False,
    ) -> dict[int, dict]:
        """
        BALANCED row-block messages:

        - For splittable tensors:
            split into d row-blocks (d = sender degree),
            shuffle blocks deterministically with seed,
            send EXACTLY one block to each neighbor.
            each neighbor sees approximately 1/d of those tensors every round.

        - For unsplittable tensors:
            if broadcast_unsplittable=True: send to all neighbors
            else: send whole tensor to exactly ONE randomly chosen neighbor.

        Returns:
          dict[neighbor_id -> {"parts": [ {k,start,end,v}, ... ]}]
        """
        if not self.neighbors:
            return {}

        msg = self._get_full_or_delta_state()
        nbs = sorted(self.neighbors)
        d = len(nbs)
        rng = random.Random(seed)

        out = {nb: {"parts": []} for nb in nbs}

        g = torch.Generator(device=self.model.state_dict()[next(iter(self.model.state_dict()))].device)
        if seed is not None:
            g.manual_seed(int(seed))

        # For each tensor
        for k, t in msg.items():
            if not torch.is_tensor(t):
                continue

            # Skip BN counters etc. will crash otherwise
            if not torch.is_floating_point(t):
                continue

            can_split = enable_chunking and (t.ndim >= 1) and (t.size(0) >= d) and (t.numel() >= split_threshold_numel)

            if can_split:
                blocks = self._split_rows(t, d)  # exactly d blocks
                rng.shuffle(blocks)              # random permutation per round (deterministic given seed)

                for nb, (s, e, chunk) in zip(nbs, blocks):
                    v = chunk
                    out[nb]["parts"].append({"k": k, "start": int(s), "end": int(e), "v": v.detach().clone()})

            else:
                part_v = t
                part = {"k": k, "start": None, "end": None, "v": part_v.detach().clone()}

                if broadcast_unsplittable or not enable_chunking:
                    for nb in nbs:
                        out[nb]["parts"].append(part)
                else:
                    # Give randomly to exactly one neighbor 
                    nb = nbs[rng.randrange(d)]
                    out[nb]["parts"].append(part)

        return out

    def average_with_neighbors(self):
        """
        Full or delta averaging with row-block payloads.

        Expected payload per neighbor:
          {"parts": [{"k": str, "start": int|None, "end": int|None, "v": Tensor}, ...]}
        """
        if not self._buffer_neighbor_models:
            return

        beta = float(self.beta)
        assert 0.0 <= beta <= 1.0

        with torch.no_grad():
            self_state = self.model.state_dict()
            new_state = {k: v.detach().clone() for k, v in self_state.items()}

            if self.message_type == "full":
                # accum[k]: per-entry weighted sum of neighbor contributions for parameter k
                accum: dict[str, torch.Tensor] = {}

                # wsum[k]: per-entry sum of weights that contributed (used for normalization)
                wsum: dict[str, torch.Tensor] = {}

                # mask[k]: boolean tensor marking which entries of k received at least one update
                mask: dict[str, torch.Tensor] = {}

                for nb_id, payload in self._buffer_neighbor_models.items():
                    if not payload:
                        continue

                    # metropolis weights initialized in topologies/topology.py
                    # For each undirected edge (i, j), assign
                    # w_ij = 1 / max(deg(i), deg(j)).
                    w_ij = float(self.neighbor_weights.get(nb_id, 0.0))
                    if w_ij == 0.0:
                        continue

                    for part in payload.get("parts", []):
                        k = part["k"]
                        if k not in new_state:
                            continue
                        # Skip BN counters etc. will crash otherwise
                        if not torch.is_floating_point(new_state[k]):
                            continue

                        v = part["v"].to(device=new_state[k].device, dtype=new_state[k].dtype)
                        s, e = part["start"], part["end"]

                        if k not in accum:
                            accum[k] = torch.zeros_like(new_state[k])
                            wsum[k]  = torch.zeros_like(new_state[k])
                            mask[k] = torch.zeros_like(new_state[k], dtype=torch.bool)

                        # Full tensor
                        if s is None or e is None:
                            accum[k] += w_ij * v
                            wsum[k]  += w_ij
                            mask[k].fill_(True)
                        # Chunk of Tensor
                        else:
                            accum[k][s:e] += w_ij * v
                            wsum[k][s:e]  += w_ij
                            mask[k][s:e] = True

                # After looping over all neighbors and all parts
                for k in accum.keys():
                    m = mask[k]
                    # new_state[k][m] = (1.0 - beta) * new_state[k][m] + beta * accum[k][m]
                    # Normalize by the sum of weights that actually contributed at each entry.
                    denom = wsum[k][m].clamp_min(1e-12)
                    new_state[k][m] = (1.0 - beta) * new_state[k][m] + beta * (accum[k][m] / denom)
            
            elif self.message_type == "delta":
                delta_accum: dict[str, torch.Tensor] = {}
                wsum: dict[str, torch.Tensor] = {}
                mask: dict[str, torch.Tensor] = {}

                for nb_id, payload in self._buffer_neighbor_models.items():
                    if not payload:
                        continue

                    w_ij = float(self.neighbor_weights.get(nb_id, 0.0))
                    if w_ij == 0.0:
                        continue

                    for part in payload.get("parts", []):
                        k = part["k"]
                        if k not in new_state:
                            continue
                        if not torch.is_floating_point(new_state[k]):
                            continue

                        dv = part["v"].to(device=new_state[k].device, dtype=new_state[k].dtype)
                        s, e = part["start"], part["end"]

                        if k not in delta_accum:
                            delta_accum[k] = torch.zeros_like(new_state[k])
                            wsum[k] = torch.zeros_like(new_state[k])
                            mask[k] = torch.zeros_like(new_state[k], dtype=torch.bool)

                        if s is None or e is None:
                            delta_accum[k] += w_ij * dv
                            wsum[k] += w_ij
                            mask[k].fill_(True)
                        else:
                            delta_accum[k][s:e] += w_ij * dv
                            wsum[k][s:e] += w_ij
                            mask[k][s:e] = True

                for k in delta_accum.keys():
                    m = mask[k]
                    denom = wsum[k][m].clamp_min(1e-12)
                    new_state[k][m] = new_state[k][m] + beta * (delta_accum[k][m] / denom)

            else:
                raise ValueError(f"Unknown message_type: {self.message_type}")

            self.model.load_state_dict(new_state)

        self._buffer_neighbor_models = {}
