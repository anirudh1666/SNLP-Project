import torch
import torch.nn as nn
import torch.nn.functional as F 
from PositionalEmbedding import PositionalEmbedding
from Encoder import Encoder
from Decoder import DecoderNE

class DecoderOnlyTransformer(nn.Module):
  def __init__(self, src_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, compressed=False):
    super().__init__()
    self._decoders = nn.ModuleList([DecoderNE(d_model, h, d_ff, dropout, compressed=compressed) for _ in range(N)])
    self._decoder_norm = nn.LayerNorm(d_model)
    self._src_pos_encoder = PositionalEmbedding(src_vocab, d_model, dropout)
    self._word_gen = nn.Linear(d_model, tgt_vocab)

    for p in self.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

  def forward(self, src, src_mask):
    dec_out = self.decode(src, src_mask)
    return F.log_softmax(self._word_gen(dec_out), dim=-1)
  
  def decode(self, src, src_mask):
    src_embed = self._src_pos_encoder(src)
    for decoder in self._decoders:
      src_embed = decoder(src_embed, src_mask)
    return self._decoder_norm(src_embed)
