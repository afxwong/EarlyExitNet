import torch
import torch.nn as nn
from EarlyStopException import EarlyExitException

class OptionalExitModule(nn.Module):

    def __init__(self, module, num_outputs, force_forward=False, force_exit=False):
        assert not (force_forward and force_exit)
        super(OptionalExitModule, self).__init__()
        self.module = module
        self.should_force_forward = force_forward
        self.should_force_exit = force_exit
        self.num_outputs = num_outputs
        self.exit_gate = None
        self.classifier = None
        self.take_exit = None
        self.exit_idx = []
        self.early_y = None

    def force_forward(self, should_force_forward=True):
        assert not (should_force_forward and self.should_force_exit)
        self.should_force_forward = should_force_forward

    def force_exit(self, should_force_exit=True):
        assert not (self.should_force_forward and should_force_exit)
        self.should_force_exit = should_force_exit

    def forward(self, X):
        # Check the device of input tensor X and move necessary components to the same device
        current_device = X.device
        if self.exit_gate is not None:
            self.exit_gate.to(current_device)
        if self.classifier is not None:
            self.classifier.to(current_device)

        self.early_y = None

        if self.should_force_forward:
            return self.module(X)

        X_flat = torch.flatten(X, start_dim=1).to(current_device)
        batch_size, flat_size = X_flat.shape

        # Create exit gate and classifier at runtime to adapt to module input size
        if self.exit_gate is None:
            self.exit_gate = nn.Linear(flat_size, 1).to(current_device)
        if self.classifier is None:
            self.classifier = nn.Linear(flat_size, self.num_outputs).to(current_device)

        if self.should_force_exit:
            self.take_exit = torch.ones((batch_size,), device=current_device)
        else:
            self.take_exit = torch.flatten(self.exit_gate(X_flat)).to(current_device)

        exit_mask = self.take_exit > 0
        self.exit_idx = torch.where(exit_mask)[0]
        num_exits = len(self.exit_idx)

        if num_exits > 0:
            X_classify = X_flat[exit_mask]
            y_classify = self.classifier(X_classify)
            self.early_y = y_classify

        if num_exits == batch_size:
            raise EarlyExitException()

        return self.module(X[~exit_mask])





