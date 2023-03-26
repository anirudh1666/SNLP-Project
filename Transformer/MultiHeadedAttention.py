import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class MultiHeadedAttention(nn.Module):
  def __init__(self, h, d_model, dropout=0.1):
    super().__init__()
    self._d_k = d_model // h
    self.h = h
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

  def forward(self, query, key, value, mask=None):
    if mask is not None:
      mask = mask.unsqueeze(1)
    nbatches = query.size(0)

    # 1) Do all the linear projections in batch from d_model => h x d_k 
    query, key, value = \
        [l(x).view(nbatches, -1, self.h, self._d_k).transpose(1, 2)
          for l, x in zip(self._projectors, (query, key, value))]
        
    # 2) Apply attention on all the projected vectors in batch. 
    x, self.attn = self._attention(query, key, value, mask=mask, 
                              dropout=self.dropout)
        
    # 3) "Concat" using a view and apply a final linear. 
    x = x.transpose(1, 2).contiguous() \
          .view(nbatches, -1, self.h * self._d_k)
    return self._projectors[-1](x)