import copy
import torch

class Node:
    def __init__(self, node_id, model_fn, dataloader, neighbors, settings, global_init):
        self.id = node_id
        self.model = model_fn().to(settings.torch_device_name)             # fresh model instance
        self.dataloader = dataloader
        self.neighbors = neighbors           # list of neighbor node_ids
        self.optimizer = self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=settings.learning.learning_rate,
            momentum=settings.learning.momentum,
            weight_decay=settings.learning.weight_decay,
        )
        self._buffer_neighbor_models = []    # for storing received models
        self.model.load_state_dict(global_init)

    # num_steps = number of gradient steps the node takes this round
    def local_train(self, local_epochs=1, device="cpu"):
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
        # state_dict() is a PyTorch nn.Module method:
        # - returns a dict: {parameter_name: tensor, ...}
        # - contains all learnable weights (and buffers) of the model
        # We deepcopy it so we can send/average weights safely between nodes
        return copy.deepcopy(self.model.state_dict())

    def receive_message(self, neighbor_state_dict):
        self._buffer_neighbor_models.append(neighbor_state_dict)

    def average_with_neighbors(self):
        if not self._buffer_neighbor_models:
            return  # nothing to average

        # include own parameters in the average
        all_states = [self.model.state_dict()] + self._buffer_neighbor_models
        avg_state = {}

        # assuming all_states have same keys (iterating over all keys)
        # Will need to change if we implement chunking will nog longer match
        for key in all_states[0].keys():
            avg_param = sum(state[key] for state in all_states) / len(all_states)
            avg_state[key] = avg_param

        self.model.load_state_dict(avg_state)
        self._buffer_neighbor_models = []
