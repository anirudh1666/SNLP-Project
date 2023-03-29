import torch 
import torch.nn as nn
from multiheadedattention import MultiHeadedAttention
from feedforward import FeedForward

class Decoder(nn.Module):
  def __init__(self, d_model, h, d_ff, dropout):
    super().__init__()
    self._self_attn = MultiHeadedAttention(h, d_model)
    self._src_attn = MultiHeadedAttention(h, d_model)
    self._ff = FeedForward(d_model, d_ff, dropout)
    self._norm1 = nn.LayerNorm(d_model)
    self._norm2 = nn.LayerNorm(d_model)
    self._norm3 = nn.LayerNorm(d_model)
    self._dropout1 = nn.Dropout(dropout)
    self._dropout2 = nn.Dropout(dropout)

  def forward(self, x, encoder_output, src_mask, tgt_mask):
    x2 = self._dropout1(self._self_attn(x, x, x, tgt_mask))
    x = self._norm1(x + x2)
    x2 = self._dropout2(self._src_attn(x, encoder_output, encoder_output, src_mask))
    x = self._norm2(x + x2)
    return self._norm3(x + self._ff(x))
  
  
class DecoderNE(nn.Module):
  def __init__(self, d_model, h, d_ff, dropout):
    super().__init__()
    self._self_attn_1 = MultiHeadedAttention(h, d_model)
    self._self_attn_2 = MultiHeadedAttention(h, d_model)
    self._ff = FeedForward(d_model, d_ff, dropout)
    self._norm1 = nn.LayerNorm(d_model)
    self._norm2 = nn.LayerNorm(d_model)
    self._norm3 = nn.LayerNorm(d_model)
    self._dropout1 = nn.Dropout(dropout)
    self._dropout2 = nn.Dropout(dropout)

  def forward(self, x, tgt_mask):
    x2 = self._dropout1(self._self_attn_1(x, x, x, tgt_mask))
    x = self._norm1(x + x2)
    x2 = self._dropout2(self._src_attn_2(x, x, x))  # does this need mask?
    x = self._norm2(x + x2)
    return self._norm3(x + self._ff(x))
