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
        try:
            last_layer_y_hat = self.model(X)
        except Exception as e:
            if str(e).startswith('All Early Exited'):
                pass
            else:
                raise e
        return last_layer_y_hat



