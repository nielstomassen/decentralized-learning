import copy
import torch

class Node:
    def __init__(self, node_id, model_fn, dataloader, neighbors, settings, global_init, neighbor_weights):
        self.id = node_id
        self.model = model_fn().to(settings.torch_device_name)             # fresh model instance
        self.dataloader = dataloader
        self.neighbors = neighbors           # list of neighbor node_ids
        self.neighbor_weights = neighbor_weights  # dict {node_id: weight}
        self.beta = settings.beta
        self._state_before_local = None
        self.message_type = settings.message_type  # "full" or "delta"

        self.optimizer = self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=settings.learning.learning_rate,
            momentum=settings.learning.momentum,
            weight_decay=settings.learning.weight_decay,
        )
        self._buffer_neighbor_models = {}    # for storing received models
        self.model.load_state_dict(global_init)      

    # num_steps = number of gradient steps the node takes this round
    def local_train(self, local_epochs=1, device="cpu"):
        self._state_before_local = copy.deepcopy(self.model.state_dict())
        # Put model in training mode
        self.model.train()
        # Iterator over dataloader (so we can call next())
        data_iter = iter(self.dataloader)

        num_batches = len(self.dataloader)
        total_steps = local_epochs * num_batches
        for _ in range(total_steps):
            try:
                x, y = next(data_iter)
            # If dataloader is exhausted
            except StopIteration:
                # Create new iterator: cycle through batches again
                # Only happens if num_steps > number of batches
                data_iter = iter(self.dataloader)
                x, y = next(data_iter)

            # Put the image batch x and labels y on CPU or GPU.
            x, y = x.to(device), y.to(device)
            
            # Classic pytorch training step
            self.optimizer.zero_grad() # Clears old gradients from the previous step
            
            # Passes the batch through the model (model.forward(x)).
            # Output is tensor of raw class scores 
            # logits shape: [batch_size, num_classes].
            # these are NOT probabilities; cross_entropy() will apply softmax internally
            logits = self.model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            # Computes gradients of loss w.r.t. all trainable parameters.
            loss.backward()
            # Updates parameters using the chosen optimization algorithm
            self.optimizer.step()

    
    def prepare_message(self):
        cur = self.model.state_dict()

        if self.message_type == "full":
            return copy.deepcopy(cur)

        elif self.message_type == "delta":
            if self._state_before_local is None:
                raise RuntimeError("Delta message requested but no pre-local snapshot found.")
            delta = {}
            for k in cur.keys():
                delta[k] = cur[k] - self._state_before_local[k]
            return delta  # already new tensors; deepcopy optional
        else:
            raise ValueError(f"Unknown message_type: {self.message_type}")

    def receive_message(self, neighbor_id, neighbor_state_dict):
        self._buffer_neighbor_models[neighbor_id] = neighbor_state_dict

    def average_with_neighbors(self):
        if not self._buffer_neighbor_models:
            return  # nothing to average

        beta = self.beta
        assert 0.0 <= beta <= 1.0

        with torch.no_grad():
            self_state = self.model.state_dict()
            new_state = {}
            # keys of the parameter tensors
            keys = self_state.keys()

            if self.message_type == "full":
                for k in keys:
                    # 1) neighbor mixture: sum_j W_ij * theta_j[k]
                    neighbor_mix = 0.0
                    for nb_id, nb_state in self._buffer_neighbor_models.items():
                        # Get nb weight or 0.0 if it doesnt exist (ignore it)
                        w_ij = self.neighbor_weights.get(nb_id, 0.0)
                        neighbor_mix = neighbor_mix + w_ij * nb_state[k]

                    # 2) convex combination with self
                    new_state[k] = (1.0 - beta) * self_state[k] + beta * neighbor_mix
                
            elif self.message_type == "delta":
                # Delta rule: theta <- theta + beta * sum_j w_ij * delta_j
                for k in keys:
                    delta_mix = 0.0
                    for nb_id, nb_delta in self._buffer_neighbor_models.items():
                        w_ij = self.neighbor_weights.get(nb_id, 0.0)
                        delta_mix = delta_mix + w_ij * nb_delta[k]

                    new_state[k] = self_state[k] + beta * delta_mix

            else:
                raise ValueError(f"Unknown message_type: {self.message_type}")

            self.model.load_state_dict(new_state)
        self._buffer_neighbor_models = {}