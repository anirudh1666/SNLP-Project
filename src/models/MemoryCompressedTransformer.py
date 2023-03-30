import torch
import torch.nn as nn
import torch.nn.functional as F 
from utils.transformer.positional_embedding import PositionalEmbedding
from utils.transformer.encoder import Encoder
from utils.transformer.decoder import Decoder

class MemoryCompressedTransformer(nn.Module):
  def __init__(self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    super().__init__()
    self._encoders = nn.ModuleList([Encoder(dropout, h, d_model, d_ff, compressed=True) for _ in range(N)])
    self._encoder_norm = nn.LayerNorm(d_model)
    self._decoders = nn.ModuleList([Decoder(d_model, h, d_ff, dropout, compressed=True) for _ in range(N)])
    self._decoder_norm = nn.LayerNorm(d_model)
    self._src_pos_encoder = PositionalEmbedding(src_vocab, d_model, dropout)
    self._tgt_pos_encoder = PositionalEmbedding(tgt_vocab, d_model, dropout)

    self._word_gen = nn.Linear(d_model, tgt_vocab)

    for p in self.parameters():
        if p.dim() > 1:
          
            nn.init.xavier_uniform(p)

  def forward(self, src, tgt, src_mask, tgt_mask):
    enc_out = self.encode(src, src_mask)
    dec_out = self.decode(enc_out, src_mask, tgt, tgt_mask)

    return F.log_softmax(self._word_gen(dec_out), dim=-1)
  
  def encode(self, src, src_mask):
    src_embed = self._src_pos_encoder(src)
    for encoder in self._encoders:
      src_embed = encoder(src_embed, src_mask)
    return self._encoder_norm(src_embed)
  
  def decode(self, enc_out, src_mask, tgt, tgt_mask):
    tgt_embed = self._tgt_pos_encoder(tgt)
    for decoder in self._decoders:
      tgt_embed = decoder(tgt_embed, enc_out, src_mask, tgt_mask)
    return self._decoder_norm(tgt_embed)