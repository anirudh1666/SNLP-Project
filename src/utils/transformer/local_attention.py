import torch 
import torch.nn as nn 
import torch.nn.functional as F
from mhcompressedattention import MemoryCompressedAttention
from multiheadedattention import MultiHeadedAttention


class LocalAttention(nn.Module):
    def __init__(self, h, d_model, split, compressed=False, dropout=0.1):
        super(LocalAttention, self).__init__()
        self._split = split
        if not compressed:
            self._attn_mechs = nn.ModuleList([MultiHeadedAttention(h, d_model) for _ in range(split)])
        else:
            self._attn_mechs = nn.ModuleList([MemoryCompressedAttention(h, d_model) for _ in range(split)])  

    def forward(self, x):
        b, l, _ = x.shape
        assert l % self._split == 0, "sequence length not divisible by split size"

        x = x.chunk(self._split, dim=-2)
        out = torch.cat([f(c, c, c) for f, c in zip(self._attn_mechs, x)], dim=-2)
        return out
