import torch
from torch import nn
import torch.nn.functional as F


class MemoryCompress(nn.Module):
    def __init__(self, d_model, ratio, groups = 1):
        super(MemoryCompress, self).__init__()
        self.conv = nn.Conv1d(d_model, d_model, ratio, stride = ratio, groups = groups)

    def forward(self, x):
        x = x.transpose(1, 2)
        compressed_x = self.conv(x)
        return compressed_x.transpose(1, 2)
