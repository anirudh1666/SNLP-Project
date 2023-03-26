import torch 
import torch.nn as nn
import torch.nn.functional as F 

class FeedForward(nn.Module):
  def __init__(self, d_model, d_ff, dropout=0.1):
    super().__init__()
    self._w1 = nn.Linear(d_model, d_ff)
    self._w2 = nn.Linear(d_ff, d_model)
    self._dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self._w2(self._dropout(F.relu(self._w1(x))))