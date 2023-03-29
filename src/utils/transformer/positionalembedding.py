import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class PositionalEmbedding(nn.Module):
  def __init__(self, src_vocab, d_model, dropout, max_len=5000):
    super().__init__()
    self._embeddings = nn.Embedding(src_vocab, d_model)
    self._d_model = d_model
    self._dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = math.sqrt(self._d_model) * self._embeddings(x)
    x += Variable(self.pe[:, :x.size(1)], requires_grad=False)
    return self._dropout(x)
