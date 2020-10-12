import torch
import torch.nn as nn
import torch.nn.functional as F


# The proposed probabilistic loss for pNCNN
class MaskedProbLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, means, targets, cout):

        res = cout
        regl = torch.log(cout+1e-16)  # Regularization term

        # Pick only valid pixels
        valid_mask = (targets > 0).detach()
        targets = targets[valid_mask]
        means = means[valid_mask]
        res = res[valid_mask]
        regl = regl[valid_mask]

        loss = torch.mean(res * torch.pow(targets - means, 2) - regl)
        return loss
