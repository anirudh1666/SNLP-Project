import torch 
import torch.nn as nn 
from MultiHeadedAttention import MultiHeadedAttention
from FeedForward import FeedForward

class Encoder(nn.Module):
  def __init__(self, dropout, h, d_model, d_ff):
    super().__init__()
    self._attn = MultiHeadedAttention(h, d_model)
    self._ff = FeedForward(d_model, d_ff, dropout)
    self._norm1 = nn.LayerNorm(d_model)
    self._norm2 = nn.LayerNorm(d_model)

  def forward(self, x, mask):
    x2 = self._attn(x, x, x, mask)
    x = self._norm1(x + x2) # residual + layer norm
    return self._norm2(x + self._ff(x))