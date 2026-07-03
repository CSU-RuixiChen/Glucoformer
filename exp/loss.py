import torch
import torch.nn as nn

class MultiHorizonMSELoss(nn.Module):
    def __init__(self, pred_len, mode='power', alpha=0.1, beta=0.7):
        super().__init__()
        self.pred_len = pred_len

        h = torch.arange(1, pred_len + 1).float()

        if mode == 'uniform':
            weights = torch.ones_like(h)

        elif mode == 'decay':
            weights = torch.exp(-alpha * h)

        elif mode == 'linear':
            weights = h

        elif mode == 'power': 
            weights = h ** beta

        else:
            raise ValueError("Unknown mode")

        weights = weights / weights.sum()
        self.register_buffer('weights', weights)

    def forward(self, pred, target):
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
            target = target.squeeze(-1)

        loss = (pred - target) ** 2
        weights = self.weights.to(loss.device) 
        loss = loss * weights 

        return loss.mean()