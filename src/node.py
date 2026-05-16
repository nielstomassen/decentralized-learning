import copy
import random
from typing import Optional

import torch
import math
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager



class Node:
    """
    Node with:
      - local SGD training
      - chunked communication, either:
          * ``topology_rowblocks`` (default): per-tensor row blocks split by degree; policy controls per-neighbor vs broadcast-same;
          * ``topology_flat_degree``: flatten float weights, split into d=degree contiguous chunks (same layout as standard_chunking);
          * ``standard_chunking``: flatten, K global contiguous chunks, same subset to all neighbors
      - averaging that updates only the entries actually received (row slices or ``param_flat`` slices)
    """

    def __init__(self, node_id, model_fn, dataloader, neighbors, settings, global_init, neighbor_weights):
        self.id = node_id
        self.model = model_fn().to(settings.torch_device_name)
        self.dataloader = dataloader

        self.neighbors = list(neighbors)  # list[int]
        self.neighbor_weights = dict(neighbor_weights)  # dict[int -> float]
        
        self.beta = float(settings.beta)
        self.message_type = settings.message_type  # "full" or "delta"
        self.enable_chunking = getattr(settings, "enable_chunking", False)
        self.chunking_mode = getattr(settings, "chunking_mode", "topology_rowblocks")
        _gk = getattr(settings, "standard_chunking_global_k", None)
        self.standard_chunking_global_k = (
            int(_gk) if _gk is not None else int(max(8, int(settings.participants)))
        )
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

    def enable_dp_training(
        self,
        noise_multiplier: float,
        max_grad_norm: float,
        delta: float = 1e-5,
        physical_batch_size: Optional[int] = None,
        logical_batch_size: Optional[int] = None,
    ):
        use_memory_manager = (
            logical_batch_size is not None
            and physical_batch_size is not None
            and logical_batch_size > physical_batch_size
        )
        if use_memory_manager:
            old_loader = self.dataloader
            logical_loader = DataLoader(
                old_loader.dataset,
                batch_size=logical_batch_size,
                shuffle=True,
                drop_last=old_loader.drop_last,
                num_workers=old_loader.num_workers,
                collate_fn=old_loader.collate_fn,
            )
            data_loader_for_dp = logical_loader
        else:
            data_loader_for_dp = self.dataloader

        pe = PrivacyEngine(secure_mode=False)
        self.model, self.optimizer, self.dataloader = pe.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=data_loader_for_dp,
            noise_multiplier=float(noise_multiplier),
            max_grad_norm=float(max_grad_norm),
            poisson_sampling=False,
        )
        self.privacy_engine = pe
        self.dp_delta = float(delta)
        if use_memory_manager:
            self._dp_physical_batch_size = physical_batch_size
        else:
            self._dp_physical_batch_size = None

    def move_to_device(self, device: str) -> None:
        """Move model and optimizer state to device (for streaming: only one node on GPU at a time)."""
        self.model.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    def get_epsilon(self):
        if self.privacy_engine is None:
            return None
        if True:
            return None
        try:
            return float(self.privacy_engine.get_epsilon(delta=self.dp_delta))
        except Exception as e:
            # PRV accountant can allocate huge arrays or overflow for extreme (sigma, steps) combinations
            import warnings
            warnings.warn(
                f"get_epsilon() failed ({type(e).__name__}), returning None. "
                "Training is unchanged; only epsilon reporting is skipped.",
                UserWarning,
                stacklevel=2,
            )
            return None

    def local_train(self, local_epochs=1, device="cpu"):
        if self.message_type == "delta":
            sd = self.model.state_dict()
            # Save stated before training to compute delta: state after training - state before training
            self._state_before_local = {k: v.detach().clone() for k, v in sd.items()}
        else:
            self._state_before_local = None

        self.model.train()

        physical_batch = getattr(self, "_dp_physical_batch_size", None)
        if self.privacy_engine is not None and physical_batch is not None:
            # Opacus BatchMemoryManager: iterate in small physical batches, optimizer steps on logical batch
            for _ in range(local_epochs):
                with BatchMemoryManager(
                    data_loader=self.dataloader,
                    max_physical_batch_size=physical_batch,
                    optimizer=self.optimizer,
                ) as new_data_loader:
                    for x, y in new_data_loader:
                        x, y = x.to(device), y.to(device)
                        self.optimizer.zero_grad()
                        logits = self.model(x)
                        loss = torch.nn.functional.cross_entropy(logits, y)
                        loss.backward()
                        self.optimizer.step()
            return

        data_iter = iter(self.dataloader)
        num_batches = len(self.dataloader)
        total_steps = local_epochs * num_batches

        for _ in range(total_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
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
            # State after training - state before training (align device: snapshot may be on GPU, cur on CPU after move_to_device)
            return {k: (cur[k] - self._state_before_local[k].to(cur[k].device)).detach().clone() for k in cur.keys()}

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

    @staticmethod
    def _partition_index_range(n: int, K: int) -> list[tuple[int, int]]:
        """Partition indices [0, n) into contiguous intervals; at most min(K, max(1,n)) intervals, sizes differ by at most one."""
        if n <= 0:
            return []
        K_eff = max(1, min(int(K), n))
        base = n // K_eff
        rem = n % K_eff
        ranges: list[tuple[int, int]] = []
        s = 0
        for i in range(K_eff):
            length = base + (1 if i < rem else 0)
            e = s + length
            ranges.append((s, e))
            s = e
        return ranges

    @staticmethod
    def _parts_for_global_flat_interval(
        vec_1d: torch.Tensor,
        gs: int,
        ge: int,
        pieces: list[dict],
    ) -> list[dict]:
        """Map a global flat interval [gs, ge) into per-parameter flat slices (layout param_flat)."""
        parts: list[dict] = []
        for p in pieces:
            a = max(gs, p["g0"])
            b = min(ge, p["g1"])
            if a < b:
                lo = a - p["g0"]
                hi = b - p["g0"]
                parts.append(
                    {
                        "layout": "param_flat",
                        "k": p["k"],
                        "start": int(lo),
                        "end": int(hi),
                        "v": vec_1d[a:b].detach().clone(),
                    }
                )
        return parts

    @staticmethod
    def _clone_parts(parts: list[dict]) -> list[dict]:
        out: list[dict] = []
        for p in parts:
            q = dict(p)
            q["v"] = p["v"].detach().clone()
            out.append(q)
        return out

    def prepare_messages_flat_chunks(
        self,
        seed: int | None = None,
        enable_chunking: bool = False,
        chunks_per_neighbor: int = 1,
        num_partitions: int | None = None,
        neighbor_policy: str = "broadcast_same",
    ) -> dict[int, dict]:
        """
        Flatten all floating-point message tensors into one vector (fixed key order), partition into
        contiguous segments, assign segments to neighbors.

        - ``num_partitions``: number of equal segments; ``None`` uses sender degree ``d``.
        - ``neighbor_policy``: ``broadcast_same`` sends the same sampled segments to every neighbor;
          ``per_neighbor`` uses a sliding window over shuffled segments (edge-specific).
        """
        if not self.neighbors:
            return {}

        if not enable_chunking:
            return self.prepare_messages_for_neighbors_rowblocks(
                seed=seed,
                enable_chunking=False,
                chunks_per_neighbor=chunks_per_neighbor,
            )

        msg = self._get_full_or_delta_state()
        items = [
            (k, t)
            for k, t in msg.items()
            if torch.is_tensor(t) and torch.is_floating_point(t)
        ]
        if not items:
            return self.prepare_messages_for_neighbors_rowblocks(
                seed=seed,
                enable_chunking=False,
                chunks_per_neighbor=chunks_per_neighbor,
            )

        vec = torch.cat([t.detach().reshape(-1) for _, t in items], dim=0)
        n = int(vec.numel())
        pieces: list[dict] = []
        offset = 0
        for k, t in items:
            m = int(t.numel())
            pieces.append({"k": k, "g0": offset, "g1": offset + m})
            offset += m

        nbs = sorted(self.neighbors)
        d = len(nbs)
        K_part = int(num_partitions) if num_partitions is not None else d
        ranges = self._partition_index_range(n, K_part)
        rng = random.Random(seed)
        blocks = list(ranges)
        rng.shuffle(blocks)

        chunks_per_nb = max(1, min(int(chunks_per_neighbor), d, len(blocks)))

        if neighbor_policy == "broadcast_same":
            cap = chunks_per_nb
            chosen = rng.sample(range(len(blocks)), cap) if cap < len(blocks) else list(range(len(blocks)))
            base_parts: list[dict] = []
            for idx in sorted(chosen):
                gs, ge = blocks[idx]
                base_parts.extend(self._parts_for_global_flat_interval(vec, gs, ge, pieces))
            cloned = self._clone_parts(base_parts)
            return {nb: {"parts": cloned} for nb in nbs}

        out = {nb: {"parts": []} for nb in nbs}
        for j, nb in enumerate(nbs):
            for offset_i in range(chunks_per_nb):
                gs, ge = blocks[(j + offset_i) % len(blocks)]
                out[nb]["parts"].extend(
                    self._parts_for_global_flat_interval(vec, gs, ge, pieces)
                )
        for nb in nbs:
            out[nb]["parts"] = self._clone_parts(out[nb]["parts"])
        return out

    def prepare_messages_standard_flat_chunks(
        self,
        seed: int | None = None,
        enable_chunking: bool = False,
        chunks_per_neighbor: int = 1,
        global_k: int | None = None,
    ) -> dict[int, dict]:
        """
        **standard_chunking** baseline: global flat vector, ``K`` partitions, broadcast same subset to all neighbors.
        """
        K = int(global_k) if global_k is not None else int(self.standard_chunking_global_k)
        return self.prepare_messages_flat_chunks(
            seed=seed,
            enable_chunking=enable_chunking,
            chunks_per_neighbor=chunks_per_neighbor,
            num_partitions=K,
            neighbor_policy="broadcast_same",
        )

    def prepare_messages_topology_flat_degree(
        self,
        seed: int | None = None,
        enable_chunking: bool = False,
        chunks_per_neighbor: int = 1,
        neighbor_policy: str = "per_neighbor",
    ) -> dict[int, dict]:
        """
        **topology_flat_degree**: like standard_chunking (flatten then contiguous segments) but
        ``num_partitions = degree``; neighbor assignment from ``neighbor_policy`` (default per-neighbor).
        """
        return self.prepare_messages_flat_chunks(
            seed=seed,
            enable_chunking=enable_chunking,
            chunks_per_neighbor=chunks_per_neighbor,
            num_partitions=None,
            neighbor_policy=neighbor_policy,
        )

    def prepare_messages_for_neighbors_rowblocks(
        self,
        seed: int | None = None,
        split_threshold_numel: int = 1,
        broadcast_unsplittable: bool = False,
        enable_chunking: bool = False,
        chunks_per_neighbor: int = 1,
        neighbor_policy: str = "per_neighbor",
    ) -> dict[int, dict]:
        """
        **topology_rowblocks** (proposed / topology-aware): balanced per-tensor row-block messages.

        - For splittable tensors:
            split into d row-blocks (d = sender degree),
            shuffle blocks deterministically with seed,
            then either:
            * ``neighbor_policy="per_neighbor"`` (default): each neighbor receives up to ``chunks_per_neighbor``
              chunks via a sliding window over the shuffled blocks (**edge-specific**).
            * ``neighbor_policy="broadcast_same"``: sample ``K = max(1, min(chunks_per_neighbor, d))`` block indices
              once (without replacement among the d blocks) and append those same row-blocks to every neighbor's
              message (same traffic pattern as **standard_chunking** broadcast, but partitions follow degree-based rows).

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

                K = max(1, min(chunks_per_neighbor, d))
                if neighbor_policy == "broadcast_same":
                    chosen = rng.sample(range(d), K) if K < d else list(range(d))
                    for idx in sorted(chosen):
                        s, e, chunk = blocks[idx]
                        for nb in nbs:
                            out[nb]["parts"].append(
                                {"k": k, "start": int(s), "end": int(e), "v": chunk.detach().clone()}
                            )
                else:
                    # per_neighbor: sliding window (edge-specific chunks)
                    for j, nb in enumerate(nbs):
                        for offset in range(K):
                            idx = (j + offset) % d
                            s, e, chunk = blocks[idx]
                            out[nb]["parts"].append({"k": k, "start": int(s), "end": int(e), "v": chunk.detach().clone()})

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
                        layout = part.get("layout", "rows")

                        if k not in accum:
                            accum[k] = torch.zeros_like(new_state[k])
                            wsum[k]  = torch.zeros_like(new_state[k])
                            mask[k] = torch.zeros_like(new_state[k], dtype=torch.bool)

                        # Standard chunking
                        if layout == "param_flat":
                            lo, hi = int(part["start"]), int(part["end"])
                            flat = new_state[k].view(-1)
                            vv = v.reshape(-1)
                            accum[k].view(-1)[lo:hi] += w_ij * vv
                            wsum[k].view(-1)[lo:hi] += w_ij
                            mask[k].view(-1)[lo:hi] = True
                            continue
                        
                        # Topology rowblocks
                        s, e = part["start"], part["end"]
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
                        layout = part.get("layout", "rows")

                        if k not in delta_accum:
                            delta_accum[k] = torch.zeros_like(new_state[k])
                            wsum[k] = torch.zeros_like(new_state[k])
                            mask[k] = torch.zeros_like(new_state[k], dtype=torch.bool)

    	                # Standard chunking
                        if layout == "param_flat":
                            lo, hi = int(part["start"]), int(part["end"])
                            dvv = dv.reshape(-1)
                            delta_accum[k].view(-1)[lo:hi] += w_ij * dvv
                            wsum[k].view(-1)[lo:hi] += w_ij
                            mask[k].view(-1)[lo:hi] = True
                            continue

                        # Topology rowblocks
                        s, e = part["start"], part["end"]
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
