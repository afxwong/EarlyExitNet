import torch
import torch.nn as nn

from EarlyStopException import EarlyExitException
from OptionalExitModule import OptionalExitModule, TrainingState


class EarlyExitModel(nn.Module):

    def __init__(self, model, num_outputs, device):
        super(EarlyExitModel, self).__init__()
        self.model = model
        self.num_outputs = num_outputs
        self.exit_modules = []
        self.original_modules = {}
        self.device = device
        
        self.state = TrainingState.INFER
        self.num_exits_per_module = []
        self.costs = self.compute_costs_per_exit_module()
        
    def set_state(self, state):
        self.state = state
        for module in self.exit_modules:
            module.set_state(state)
         
    # MARK: - Compute Costs   
    def compute_costs_per_exit_module(self):
        if len(self.exit_modules) == 0:
            return torch.tensor([0.0], device=self.device)

        # Total number of parameters in the model
        total_param_count = sum(p.numel() for p in self.model.parameters())

        # Initialize lists to store costs and running parameter count
        costs = []
        running_param_count = 0.0

        # Iterate through model layers and compute cost per layer
        costs, running_param_count = self.find_exit_layers(self.model, costs, running_param_count)

        # Append the final cost, which represents the entire model
        costs.append(total_param_count)
        cost_tensor = torch.tensor(costs, device=self.device)
        cost_tensor = (cost_tensor - cost_tensor[0]) / (cost_tensor[-1] - cost_tensor[0])
        return cost_tensor

    def find_exit_layers(self, model, costs, running_param_count):
        for layer in model.children():
            if isinstance(layer, OptionalExitModule):
                # Add the parameters of the exit layer's original module rather than the exit layer itself
                running_param_count += sum(p.numel() for p in layer.module.parameters())
                costs.append(running_param_count)
            elif isinstance(layer, nn.Module):
                costs, running_param_count = self.find_exit_layers(layer, costs, running_param_count)
            else:
                running_param_count += sum(p.numel() for p in layer.parameters())
        return costs, running_param_count

    # MARK: - Add/Remove Exit Layers
    def get_attribute(self, attr):
        layers = attr.split(".")
        val = self.model
        for layer in layers:
            val = getattr(val, layer)
        return val
        
    def set_attribute(self, attr, value):
        layers = attr.split(".")
        val = self.model
        for layer in layers[:-1]:
            val = getattr(val, layer)
        setattr(val, layers[-1], value)
    
    
    def clear_exits(self):
        # replaces wrapped modules with original module
        for attr, original_module in self.original_modules.items():
            self.set_attribute(attr, original_module)
        self.original_modules = {}
        self.exit_modules = []
        self.costs = self.compute_costs_per_exit_module()

    def add_exit(self, attr):
        # add an early exit module
        layer = self.get_attribute(attr)
        self.original_modules[attr] = layer
        optional_exit_module = OptionalExitModule(layer, self.num_outputs)
        optional_exit_module.set_state(self.state)
        
        self.set_attribute(attr, optional_exit_module)
        self.exit_modules.append(optional_exit_module)
        self.costs = self.compute_costs_per_exit_module()
        return optional_exit_module

    def forward(self, X):
        batch_size, *sample_shape = X.shape
        
        terminal_layer_y_hat = None
        
        # y_hat of layer where there are no remaining images to push forward in the model. All samples have exited
        early_exit_y_hat = None
        try:
            terminal_layer_y_hat = self.model(X)
        except EarlyExitException as e:
            early_exit_y_hat = e.y_hat
        
        if self.state == TrainingState.TRAIN_CLASSIFIER_EXIT or self.state == TrainingState.TRAIN_CLASSIFIER_FORWARD:
            if terminal_layer_y_hat is not None:
                # if forward pass made it to the back of the layer, get the last layer y_hat
                return terminal_layer_y_hat
            
            # if exit occured before last classifier, get y_hat where exit occured
            return early_exit_y_hat

        elif self.state == TrainingState.TRAIN_EXIT:
            assert terminal_layer_y_hat is not None, "Forward propagation should have made it to the end of the model"
            
            y_hats = torch.empty((batch_size, len(self.exit_modules) + 1, self.num_outputs), device=self.device)
            exit_confidences = torch.empty((batch_size, len(self.exit_modules)), device=self.device)
            
            for i, exit_module in enumerate(self.exit_modules):
                y_hats[:, i] = exit_module.y_hat
                exit_confidences[:, i] = exit_module.exit_confidences
                
            y_hats[:, -1] = terminal_layer_y_hat
            
            return y_hats, exit_confidences
        
        else:
            if torch.cuda.is_available() and len(self.exit_modules) > 0:
                for stream in [module.stream for module in self.exit_modules if module.stream is not None]:
                    stream.synchronize() # wait for all multithreaded classifiers to finish
            
            y_hat = torch.empty((batch_size, self.num_outputs), device=self.device)
            
            remaining_idx = torch.arange(batch_size, device=self.device)
            
            self.num_exits_per_module = []
            
            for exit_module in self.exit_modules:
                if len(remaining_idx) == 0:
                    self.num_exits_per_module.append(0)
                    continue

                self.num_exits_per_module.append(exit_module.exit_taken_idx.sum().item())
                
                # use indices of exits taken in the model's (reduced) batched to obtain the translated original index within the original batch
                original_idx = remaining_idx[exit_module.exit_taken_idx]
                if len(original_idx) == 0: continue
                
                y_hat[original_idx] = exit_module.y_hat
                remaining_idx = remaining_idx[~exit_module.exit_taken_idx]
                
            # if even after going through each early exit layer, there are samples that did not exit, grab the y_hat from terminal classifier
            if len(remaining_idx) > 0:
                y_hat[remaining_idx] = terminal_layer_y_hat
                self.num_exits_per_module.append(len(remaining_idx))
            else:
                self.num_exits_per_module.append(0)
            
            return y_hat
