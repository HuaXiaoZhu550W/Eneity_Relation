import torch
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, labels, mask):
        loss = self.loss_fn(preds, labels)
        loss = loss * mask.float().unsqueeze(-1)
        loss = torch.sum(loss.mean(dim=2), dim=-1) / (mask.sum() + 1e-10)
        return loss.mean()
