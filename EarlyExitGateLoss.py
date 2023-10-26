import torch
import torch.nn as nn

class EarlyExitGateLoss(nn.Module):
    
    def __init__(self, device, alpha=0.5):
        super(EarlyExitGateLoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.CELoss = nn.CrossEntropyLoss()
    
    def forward(self, ys, y_hats, exit_confidences, costs):
        # ys: (batch_size, )
        # y_hats: (batch_size, num_exits+1, num_outputs)
        # exit_confidences: (batch_size, num_exits)
        # costs: (num_exits+1, )
        
        batch_size, num_classifiers, num_outputs = y_hats.shape
        num_exits = num_classifiers - 1
        
        gate_summation = torch.zeros(1, device=self.device)[0]
        
        for b in range(batch_size):
            # for each sample in the batch, get the:
            y = ys[b]  # true label
            y_hat = y_hats[b]  # predicted label distribution
            g = exit_confidences[b]  # exit confidence per exit gate
            g_hat = 1 - g  # forward confidence per exit gate
            
            # accumulate the forward confidence up to the current exit gate
            prob_reaches_gate = 1
            
            for i in range(num_exits):
                # cross entropy loss due to classification setting
                ce = self.CELoss(y_hat[i], y[i])
                # accumulate the expected loss by considering the probability of the event
                summation += prob_reaches_gate * g[i] * ce
                # update the probability of reaching the next gate by the forward confidence
                prob_reaches_gate *= g_hat[i]
            # at the terminal classifier, accumulate the CE loss where every other gate does not exit 
            gate_summation += self.CELoss(y_hat[-1], y[-1]) * prob_reaches_gate
            
        
        exit_costs = torch.zeros(1, device=self.device)[0]
            
        for b in range(batch_size):
            g = exit_confidences[b]  # exit confidence per exit gate for the given sample
            for i, exit_g in enumerate(g):
                # find the first gate, from left to right, at which an exit is taken
                if exit_g > 0.5:
                    # accumulate the experienced cost of this classification execution
                    exit_costs += costs[i]
                    break
            else:
                # if the loop doesn't break, the classification makes it to the end. Thus, the final cost is experienced
                exit_costs += costs[-1]
                
        # we want to simulatenously minimize the expected CE Loss and the expected runtime cost balanced by alpha
        loss = (1 - self.alpha) * gate_summation + self.alpha * exit_costs
        
        return loss
        
            
        
            
            
         