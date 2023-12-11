import torch
import torch.nn as nn

class EarlyExitGateLoss(nn.Module):
    
    def __init__(self, device, alpha=0.5):
        super(EarlyExitGateLoss, self).__init__()
        self.device = device
    
        assert alpha >= 0 and alpha <= 1, "alpha must be between 0 and 1 (inclusive)"
        self.alpha = alpha
        
        self.CELoss = nn.CrossEntropyLoss()
    
    def forward(self, ys, y_hats, exit_confidences, costs):
        # ys: (batch_size, )
        # y_hats: (batch_size, num_exits+1, num_outputs)
        # exit_confidences: (batch_size, num_exits)
        # costs: (num_exits+1, )
        
        batch_size, num_classifiers, _ = y_hats.shape
        num_exits = num_classifiers - 1
        
        # MARK: - 1st Half of Loss Function
        gate_summation = torch.zeros(1, device=self.device)[0]
        
        for b in range(batch_size):
            # for each sample in the batch, get the:
            y = ys[b]  # true label
            y_hat = y_hats[b]  # predicted label distribution
            g = exit_confidences[b]  # exit confidence per exit gate
            g_hat = 1 - g  # forward confidence per exit gate
            
            for i in range(num_exits):
                # get the probability that the sample reaches the ith gate (i.e. does not exit at any previous gate)
                prob_reaches_gate = torch.prod(g_hat[:i])
                
                # cross entropy loss due to classification setting
                ce = self.CELoss(y_hat[i], y)
                # accumulate the expected loss by considering the probability of the event
                gate_summation += prob_reaches_gate * g[i] * ce
                
            # at the terminal classifier, accumulate the CE loss where every other gate does not exit 
            gate_summation += self.CELoss(y_hat[-1], y) * torch.prod(g_hat)

        # MARK: - 2nd Half of Loss Function
        cost_summation = torch.zeros(1, device=self.device)[0]
        
        for b in range(batch_size):
            # for each sample in the batch, get the:
            y = ys[b]  # true label
            y_hat = y_hats[b]  # predicted label distribution
            g = exit_confidences[b]  # exit confidence per exit gate
            g_hat = 1 - g  # forward confidence per exit gate
            
            for i in range(num_exits):
                # get the probability that the sample reaches the ith gate (i.e. does not exit at any previous gate)
                prob_reaches_gate = torch.prod(g_hat[:i])
                
                # cross entropy loss due to classification setting
                ce = costs[i]
                # accumulate the expected loss by considering the probability of the event
                cost_summation += prob_reaches_gate * g[i] * ce
                
            # at the terminal classifier, accumulate the CE loss where every other gate does not exit 
            cost_summation += costs[-1] * torch.prod(g_hat)
        
        # we want to simulatenously minimize the mean CE Loss and the expected runtime cost balanced by alpha
        loss = (1 - self.alpha) * (gate_summation / batch_size) + self.alpha * (cost_summation / batch_size)
        
        return loss, (1 - self.alpha) * (gate_summation / batch_size), self.alpha * (cost_summation / batch_size)
        
            
        
            
            
         