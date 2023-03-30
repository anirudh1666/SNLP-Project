import torch 
import torch.nn as nn 
from utils.transformer.multiheaded_attention import MultiHeadedAttention
from utils.transformer.mh_compressed_attention import MemoryCompressedAttention
from utils.transformer.feedforward import FeedForward

class Encoder(nn.Module):
  def __init__(self, dropout, h, d_model, d_ff, compressed=False):
    super().__init__()
    if not compressed:
      self._attn = MultiHeadedAttention(h, d_model)
    else:
      self._attn = MemoryCompressedAttention(h, d_model)
    self._ff = FeedForward(d_model, d_ff, dropout)
    self._norm1 = nn.LayerNorm(d_model)
    self._norm2 = nn.LayerNorm(d_model)
    self._dropout = nn.Dropout(dropout)

  def forward(self, x, mask):
    x2 = self._dropout(self._attn(x, x, x, mask))
    x = self._norm1(x + x2) # Norm (residual + dropout(attention(x)) )
    return self._norm2(x + self._ff(x))