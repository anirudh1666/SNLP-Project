import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.transformer.memorycompression import MemoryCompress

class MemoryCompressedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, compress_ratio=3):
        super(MemoryCompressedAttention, self).__init__()
        self._d_k = d_model // h
        self.h = h
        self._cr = compress_ratio
        self._memcomp = MemoryCompress(d_model, ratio=compress_ratio)
        self._projectors = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self._dropout = nn.Dropout(p=dropout)


    def _attention(self, query, key, value, mask=None, dropout=None):
        
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(self._d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def compress_mask(self, mask, cf, padding):
        mask = F.pad(mask.T, (0, 0, 0, padding), value=1)
        sl, emb = mask.shape
        new_mask = torch.zeros((sl//cf, emb))
        for i in range(sl//cf):
          val = torch.min(mask[i*cf:(i+1)*cf])
          new_mask[i] += val
        return new_mask.T.type(torch.bool)

    def forward(self, query, key, value, mask=None):
    
        b, l, _ = query.shape

        # 1) Compress key and values, use padding if seq len not divisible by compress ratio
        padding = self._cr - (l % self._cr)
        
        if padding < self._cr:
            key, value = map(lambda t: F.pad(t, (0, 0, padding, 0)), (key, value))

        key, value = map(self._memcomp, (key, value))

        mask = self.compress_mask(mask, self._cr, padding)
        # 2) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(b, -1, self.h, self._d_k).transpose(1, 2)
              for l, x in zip(self._projectors, (query, key, value))]
            
        # 3) Apply attention on all the projected vectors in batch. 
        x, self.attn = self._attention(query, key, value, mask=mask, 
                                  dropout=self._dropout)
            
        # 4) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
              .view(b, -1, self.h * self._d_k)
        return self._projectors[-1](x)
