
import torch
import torch.nn as nn


class ConfL2Loss(nn.Module):
    def __init__(self, args):
        super(ConfL2Loss, self).__init__()

        self.args = args
        self.t_valid = 0.0001

    def forward(self, pred, gt, cout):
        gt = torch.clamp(gt, min=0, max=self.args.max_depth)
        pred = torch.clamp(pred, min=0, max=self.args.max_depth)
        cout = torch.clamp(cout, min=0, max=self.args.max_depth)

        mask = (gt > self.t_valid).type_as(pred).detach()

        d = (1/cout * torch.pow(pred - gt, 2) - torch.log(cout)) * mask

        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask, dim=[1, 2, 3])

        loss = d / (num_valid + 1e-8)

        loss = loss.sum()

        return loss
