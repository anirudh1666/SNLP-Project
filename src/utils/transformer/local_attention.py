import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from utils.transformer.multiheaded_attention import MultiHeadedAttention


class LocalAttention(nn.Module):
    def __init__(self, h, d_model, split, dropout=0.1):
        super(LocalAttention, self).__init__()
        self._split = split
        self._attn_mechs = nn.ModuleList([MultiHeadedAttention(h, d_model) for _ in range(split)])

    def split_mask(self, length):
        shape = (1, length, length)
        subsequent_mask = np.triu(np.ones(shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

    def forward(self, x, padding_mask=None):
        b, l, _ = x.shape

        x = x.chunk(self._split, dim=-2)
        pad_masks = padding_mask.chunk(self._split, dim=-1)
        masks = [self.split_mask(c.size(-2)) for c in x]

        #print([m.shape for m in pad_masks])
        #print([m.shape for m in masks])

        #print(pad_masks[-1][0])
        #print(masks[-1][0])

        final_masks = [pm & m for pm,m in zip(pad_masks, masks)]

        #print(final_masks[-1][0])

        out = torch.cat([f(c, c, c, mask=m) for f, c, m in zip(self._attn_mechs, x, final_masks)], dim=-2)
        return out
