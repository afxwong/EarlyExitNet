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
            
    def compute_costs_per_exit_module(self):
        if len(self.exit_modules) == 0:
            return torch.tensor([1.0], device=self.device)

        # Total number of parameters in the model
        total_param_count = sum(p.numel() for p in self.model.parameters())

        # Iterate through model layers and compute cost per layer
        costs = []
        running_param_count = 0.0
        for layer in self.model.children():
            if isinstance(layer, OptionalExitModule):
                # Add the parameters of the exit layer's original module rather than the exit layer itself
                running_param_count += sum(p.numel() for p in layer.module.parameters())
                costs.append(running_param_count / total_param_count)
            else:
                running_param_count += sum(p.numel() for p in layer.parameters())

        # Append the final cost, which represents the entire model
        costs.append(1.0)
        cost_tensor = torch.tensor(costs, device=self.device)
        cost_tensor -= cost_tensor[0]
        cost_tensor /= cost_tensor[-1]
        return cost_tensor

    def clear_exits(self):
        # replaces wrapped modules with original module
        for attr, original_module in self.original_modules.items():
            setattr(self.model, attr, original_module)
        self.original_modules = {}
        self.exit_modules = []
        self.costs = self.compute_costs_per_exit_module()

    def add_exit(self, attr, model_type):
        # add an early exit module
        if model_type == 'resnet':
            layer = getattr(self.model, attr)
        else:
            if "features" in attr:
                edited_attr = attr.replace("features.", "")
                layer = getattr(self.model.features, edited_attr)
            elif 'classifier' in attr:
                edited_attr = attr.replace("classifier.", "")
                layer = getattr(self.model.classifier, edited_attr)
            else: layer = getattr(self.model, attr)
        self.original_modules[attr] = layer
        optional_exit_module = OptionalExitModule(layer, self.num_outputs)
        optional_exit_module.set_state(self.state)
        if model_type == 'resnet':
            setattr(self.model, attr, optional_exit_module)
        else:
            if "features" in attr:
                setattr(self.model.features, edited_attr, optional_exit_module)
            elif 'classifier' in attr:
                setattr(self.model.classifier, edited_attr, optional_exit_module)
            else:
                setattr(self.model, attr, optional_exit_module)
        self.exit_modules.append(optional_exit_module)
        self.costs = self.compute_costs_per_exit_module()
        return optional_exit_module

    def forward(self, X):
        
        batch_size, *sample_shape = X.shape
        
        last_layer_y_hat = None
        
        # y_hat of layer where there are no remaining images to push forward in the model. All samples have exited
        final_exit_y_hat = None
        try:
            last_layer_y_hat = self.model(X)
        except EarlyExitException as e:
            final_exit_y_hat = e.y_hat
        
        if self.state == TrainingState.TRAIN_CLASSIFIER_EXIT or self.state == TrainingState.TRAIN_CLASSIFIER_FORWARD:
    
            if last_layer_y_hat is not None:
                # if forward pass made it to the back of the layer, get the last layer y_hat
                return last_layer_y_hat
            
            # if exit occured before last classifier, get y_hat where exit occured
            return final_exit_y_hat
        
        
        if self.state == TrainingState.TRAIN_EXIT:
            
            assert last_layer_y_hat is not None, "forward propagation should have made it to the end of the model"
            
            y_hats = torch.empty((batch_size, len(self.exit_modules) + 1, self.num_outputs), device=self.device)
            exit_confidences = torch.empty((batch_size, len(self.exit_modules)), device=self.device)
            
            # TODO: set cost constant for each layer later
            
            for i, exit_module in enumerate(self.exit_modules):
                y_hats[:, i] = exit_module.y_hat
                exit_confidences[:, i] = exit_module.exit_confidences
                
            y_hats[:, -1] = last_layer_y_hat
            
            return y_hats, exit_confidences
        
        
        if self.state == TrainingState.INFER:
            
            y_hat = torch.empty((batch_size, self.num_outputs), device=self.device)
            
            remaining_idx = torch.arange(batch_size).to(self.device)
            
            self.num_exits_per_module = []
            
            for exit_module in self.exit_modules:
                try:
                    self.num_exits_per_module.append(len(exit_module.exit_taken_idx))
                except:
                    self.num_exits_per_module.append(0)
                    
                if len(remaining_idx) == 0: break
                if len(exit_module.exit_taken_idx) == 0: continue
                
                # use indices of exits taken in the model's (reduced) batched to obtain the translated original index within the original batch
                original_idx = remaining_idx[exit_module.exit_taken_idx]
                y_hat[original_idx] = exit_module.y_hat
                
                # mirroring how the batch is reduced by the exit module, reduce index look up array the same way
                to_keep = torch.ones(remaining_idx.shape)
                to_keep[exit_module.exit_taken_idx] = 0
                remaining_idx = remaining_idx[to_keep == 1]
                
            # if even after going through each early exit layer, there are samples that did not exit, grab the y_hat from terminal classifier
            if len(remaining_idx) > 0:
                y_hat[remaining_idx] = last_layer_y_hat
                self.num_exits_per_module.append(len(remaining_idx))
            
            return y_hat
