import torch
import torch.nn as nn

from EarlyStopException import EarlyExitException
from OptionalExitModule import OptionalExitModule

class EarlyExitModel(nn.Module):

    def __init__(self, model, num_outputs, device):
        super(EarlyExitModel, self).__init__()
        self.model = model
        self.num_outputs = num_outputs
        self.exit_modules = []
        self.original_modules = {}
        self.device = device


    def clear_exits(self):
        for attr, original_module in self.original_modules.items():
            setattr(self.model, attr, original_module)
        self.original_modules = {}
        self.exit_modules = []

    def add_exit(self, attr):
        layer = getattr(self.model, attr)
        self.original_modules[attr] = layer
        optional_exit_module = OptionalExitModule(layer, self.num_outputs)
        setattr(self.model, attr, optional_exit_module)
        self.exit_modules.append(optional_exit_module)
        return optional_exit_module

    def forward(self, X):
        last_layer_y_hat = None
        try:
            last_layer_y_hat = self.model(X)
        except Exception as e:
            if not isinstance(e, EarlyExitException):
                raise e
        y_hat = torch.empty((len(X), self.num_outputs), device=self.device)
        exit_gate_logits = torch.empty((len(X), 1), device=self.device)
        exit_points = torch.ones(len(X), device=self.device) * len(self.exit_modules)
        idx = torch.arange(len(X)).to(self.device)
        for i, exit_module in enumerate(self.exit_modules):
            if len(idx) == 0:
                break
            if exit_module.early_y is None:
                continue
            original_idx = idx[exit_module.exit_idx]
            y_hat[original_idx] = exit_module.early_y
            exit_gate_logits[original_idx] = exit_module.should_exit_results
            exit_points[original_idx] = i
            keep_mask = torch.ones(idx.shape, dtype=torch.bool, device=self.device)
            keep_mask[exit_module.exit_idx] = False
            idx = idx[keep_mask]
        if last_layer_y_hat is not None:
            y_hat[idx] = last_layer_y_hat
            # set infinity for exit gate logits so that they are tanh back to 1
            exit_gate_logits[idx] = float('inf')

        return y_hat, exit_points, exit_gate_logits



