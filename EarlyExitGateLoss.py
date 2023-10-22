import torch
import torch.nn as nn

class EarlyExitGateLoss(nn.Module):
    
    def forward(self, y, y_hats, exit_confidences, costs):
        # y: (batch_size, )
        # y_hats: (batch_size, num_exits+1, num_outputs)
        # exit_confidences: (batch_size, num_exits)
        # costs: (batch_size, num_exits+1)
        
        # get y_hat for whole batch but first exit
        tmp_yhat = y_hats[:, 0, :]
        
        loss = nn.CrossEntropyLoss()
        
        return -1 * torch.sum(exit_confidences)
         