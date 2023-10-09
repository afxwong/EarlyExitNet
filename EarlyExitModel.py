import torch
import torch.nn as nn

from OptionalExitModule import OptionalExitModule

class EarlyExitModel(nn.Module):

    def __init__(self, model, num_outputs):
        super(EarlyExitModel, self).__init__()
        self.model = model
        self.num_outputs = num_outputs
        self.exit_modules = []
        self.original_modules = {}


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
        y_hat = self.model(X)
        remaining_idx = torch.arange(len(X))
        exit_ids_taken = torch.ones(len(X)) * len(self.exit_modules)
        # for i, exit_module in enumerate(self.exit_modules):
        #     exit_idx = exit_module.exit_idx
        #     original_idx = remaining_idx[exit_idx]
        #     exit_ids_taken[original_idx] = i

        return y_hat, exit_ids_taken



