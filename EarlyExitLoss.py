import torch
import torch.nn as nn

class EarlyExitWeightedLoss(nn.Module):
    def __init__(self, total_exit_points):
        super().__init__()  # Use super() without specifying arguments
        self.total_exit_points = total_exit_points
        self.loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets, exit_points, shouldWeight=True):
        # Convert targets to long data type
        targets = targets.long()

        # Calculate the loss for each example
        losses = self.loss(predictions, targets)

        # Apply scaling to the loss for each example
        if shouldWeight:
            # Calculate the scaling factor for each example
            scalars = (exit_points+1) / self.total_exit_points
            
            # Multiply each example's loss by its scaling factor
            losses = losses * scalars

        # Take the mean of the weighted losses across all examples
        weighted_loss = torch.mean(losses)

        return weighted_loss
