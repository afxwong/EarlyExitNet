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
        self.should_freeze_classifier = False
       
        self.num_outputs = num_outputs
        self.exit_gate = None
        self.classifier = None
        self.take_exit = None
        self.exit_idx = []
        self.early_y = None
        self.should_exit_results = None
        
        self.optimizer = None

    def force_forward(self, should_force_forward=True):
        assert not (should_force_forward and self.should_force_exit)
        self.should_force_forward = should_force_forward

    def force_exit(self, should_force_exit=True):
        assert not (self.should_force_forward and should_force_exit)
        self.should_force_exit = should_force_exit
        
    def freeze_classifier(self, should_freeze_classifier=True):
        self.should_freeze_classifier = should_freeze_classifier
        # note: specifically not freezing the classifier params since this breaks loss.backward call
        # this is handled by zeroing out the gradients in the forward method
        
    def remove_forces(self):
        self.force_forward(False)
        self.force_exit(False)
        self.freeze_classifier(False)
        
    def optimize_gate_layer(self, y_pred, y, items_took_exit):
        # if you never used the layer, ignore
        if self.optimizer is None: return
        
        # get the relevant values to look at for this layer
        y = y[items_took_exit]
        y_pred = y_pred[items_took_exit]
        exits_taken = self.take_exit[items_took_exit].clone()
        
        # reset the gradients
        self.optimizer.zero_grad()
        
        # do loss based on whether you were correct or not
        # minimize loss for correct, maximize for incorrect -- make correct ones negative
        correct_mask = torch.argmax(y_pred, dim=1) == y
        exits_taken[correct_mask] *= -1
        
        # your loss is the sum of the exit gate values
        # more negative means more correct
        loss = exits_taken.sum()
        
        # only backprop on the exit gate to avoid double counting
        torch.autograd.backward(loss, inputs=[self.exit_gate.weight, self.exit_gate.bias])
        self.optimizer.step()
        
        # TODO: optionally scale the loss by how incorrect/correct the prediction was
        

    def forward(self, X):
        # Check the device of input tensor X and move necessary components to the same device
        current_device = X.device
        if self.exit_gate is not None:
            self.exit_gate.to(current_device)
        if self.classifier is not None:
            self.classifier.to(current_device)

        self.early_y = None
        self.should_exit_results = None

        if self.should_force_forward:
            return self.module(X)

        X_flat = torch.flatten(X, start_dim=1).to(current_device)
        batch_size, flat_size = X_flat.shape

        # Create exit gate and classifier at runtime to adapt to module input size
        if self.exit_gate is None:
            self.exit_gate = nn.Linear(flat_size, 1).to(current_device)
            self.optimizer = torch.optim.Adam(self.exit_gate.parameters(), lr=0.001)
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
            gate_results = self.take_exit[exit_mask]
            
            # store off this value for use in the loss function
            self.should_exit_results = torch.reshape(gate_results, (-1, 1))

            if self.should_freeze_classifier:
                # Use torch.no_grad() to temporarily disable gradient computation for the classifier
                with torch.no_grad():
                    y_classify = self.classifier(X_classify)
                
                # delete the grads to ensure they are not optimized when frozen
                for param in self.classifier.parameters():
                    param.grad = None
            else:
                y_classify = self.classifier(X_classify)
            self.early_y = y_classify

        if num_exits == batch_size:
            raise EarlyExitException()

        return self.module(X[~exit_mask])