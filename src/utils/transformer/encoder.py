import torch 
import torch.nn as nn 
from multiheadedattention import MultiHeadedAttention
from feedforward import FeedForward

class Encoder(nn.Module):
  def __init__(self, dropout, h, d_model, d_ff):
    super().__init__()
    self._attn = MultiHeadedAttention(h, d_model)
    self._ff = FeedForward(d_model, d_ff, dropout)
    self._norm1 = nn.LayerNorm(d_model)
    self._norm2 = nn.LayerNorm(d_model)
    self._dropout = nn.Dropout(dropout)

  def forward(self, x, mask):
    x2 = self._dropout(self._attn(x, x, x, mask))
    x = self._norm1(x + x2) # Norm (residual + dropout(attention(x)) )
    return self._norm2(x + self._ff(x))