import torch
import torch.nn as nn

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.padding_idx = padding_idx
        self.size = size
        self.smoothing = smoothing

        self._criterion = nn.KLDivLoss(size_average=False)
        self._confidence = 1 - smoothing
        self._true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self._confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self._true_dist = true_dist 

        return self._criterion(x, torch.autograd.Variable(true_dist, requires_grad=False))