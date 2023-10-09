import torch
import torch.nn as nn
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

    def force_forward(self, should_force_forward=True):
        assert not (should_force_forward and self.should_force_exit)
        self.should_force_forward = should_force_forward
    def force_exit(self, should_force_exit=True):
        assert not (self.should_force_forward and should_force_exit)
        self.should_force_exit = should_force_exit

    def forward(self, X):

        if self.should_force_forward:
            return self.module(X)

        X_flat = torch.flatten(X, start_dim=1)
        batch_size, flat_size = X_flat.shape

        # create exit gate and classifier at runtime to adapt to module input size
        if self.exit_gate is None:
            self.exit_gate = nn.Linear(flat_size, 1)
        if self.classifier is None:
            self.classifier = nn.Linear(flat_size, self.num_outputs)

        if self.should_force_exit:
            self.take_exit = torch.ones((batch_size,))
        else:
            self.take_exit = torch.flatten(self.exit_gate(X_flat))

        exit_mask = self.take_exit > 0
        self.exit_idx = torch.where(exit_mask)[0]
        num_exits = len(self.exit_idx)

        y = torch.empty((batch_size, self.num_outputs))

        if num_exits > 0:
            X_classify = X_flat[exit_mask]
            y_classify = self.classifier(X_classify)
            y[exit_mask] = y_classify

        if num_exits < batch_size:
            X_forward = X[~exit_mask]
            y_forward = self.module(X_forward)
            y[~exit_mask] = y_forward

        return y





