import torch
import torch.nn as nn
from EarlyStopException import EarlyExitException

from enum import Enum

class TrainingState(Enum):
    TRAIN_CLASSIFIER_FORWARD = 1
    TRAIN_CLASSIFIER_EXIT = 2
    TRAIN_EXIT = 3
    INFER = 4

class OptionalExitModule(nn.Module):

    def __init__(self, module, num_outputs):
        super(OptionalExitModule, self).__init__()
        
        self.module = module
        self.state = TrainingState.INFER
       
        self.num_outputs = num_outputs
        self.exit_gate = None
        self.classifier = None
        
    def set_state(self, state):
        self.state = state
          
    def forward_train_classifier_forward(self, X, X_flat):
        # roll every sample forward (entire batch) to next module
        return self.module(X)
        
    
    def forward_train_classifier_exit(self, X, X_flat):
        # every sample is forced to exit
        self.y_hat = self.classifier(X_flat)
        raise EarlyExitException(y_hat=self.y_hat)
        
        
    def forward_train_exit(self, X, X_flat):
        # get every exit confidence, every classification as if all exits were taken, and push forward in the model in the end
        X_flat = X_flat.clone().detach()
        self.exit_confidences = torch.sigmoid(torch.flatten(self.exit_gate(X_flat)))
        with torch.no_grad():
            self.y_hat = self.classifier(X_flat)
        
        return self.module(X)
    
        
    def forward_infer(self, X, X_flat):
        
        batch_size, _ = X_flat.shape
        
        # form (batch_size, ) vector of exit confidences
        self.exit_confidences = torch.sigmoid(torch.flatten(self.exit_gate(X_flat)))
        
        exit_mask = self.exit_confidences > 0.5
        self.exit_taken_idx = torch.where(exit_mask)[0]
        num_exits_taken = len(self.exit_taken_idx)

        if num_exits_taken > 0:
            X_classify = X_flat[exit_mask]
            self.y_hat = self.classifier(X_classify)

        if num_exits_taken == batch_size:
            # self.y_hat will have classificaiton results to collect later on
            raise EarlyExitException(y_hat=self.y_hat)

        return self.module(X[~exit_mask])
        
    def forward(self, X):
        
         # Check the device of input tensor X and move necessary components to the same device
        self.current_device = X.device
        
        X_flat = torch.flatten(X, start_dim=1).to(self.current_device)
        _, flat_size = X_flat.shape
        
        # Create exit gate and classifier at runtime to adapt to module input size
        if self.exit_gate is None:
            self.exit_gate = nn.Linear(flat_size, 1).to(self.current_device)
        if self.classifier is None:
            self.classifier = nn.Linear(flat_size, self.num_outputs).to(self.current_device)
        
        if self.state == TrainingState.TRAIN_CLASSIFIER_EXIT:
            return self.forward_train_classifier_exit(X, X_flat)
        
        if self.state == TrainingState.TRAIN_CLASSIFIER_FORWARD:
            return self.forward_train_classifier_forward(X, X_flat)
        
        if self.state == TrainingState.TRAIN_EXIT:
            return self.forward_train_exit(X, X_flat)
        
        if self.state == TrainingState.INFER:
            return self.forward_infer(X, X_flat)
        