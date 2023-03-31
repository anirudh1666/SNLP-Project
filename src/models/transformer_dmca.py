import torch
import torch.nn as nn
import torch.nn.functional as F 
from utils.transformer.positional_embedding import PositionalEmbedding
from utils.transformer.encoder import Encoder
from utils.transformer.decoder import DecoderNE, DecoderLocal

class TransformerDMCA(nn.Module):
  def __init__(self, src_vocab, split, compression_rate=3, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    super().__init__()
    self._decoders = self._build_module_list(N, d_model, h, d_ff, dropout, split, compression_rate)
    self._decoder_norm = nn.LayerNorm(d_model)
    self._src_pos_encoder = PositionalEmbedding(src_vocab, d_model, dropout)
    self._word_gen = nn.Linear(d_model, src_vocab)

    for p in self.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

  def forward(self, src, src_mask):
    dec_out = self.decode(src, src_mask)
    return F.log_softmax(self._word_gen(dec_out), dim=-1)
  
  def decode(self, src, src_mask):
    src_embed = self._src_pos_encoder(src)
    for decoder in self._decoders:
      src_embed = decoder(src_embed, src_mask)
    return self._decoder_norm(src_embed)

  def _build_module_list(self, N, d_model, h, d_ff, dropout, split, compression_rate):
      
    modules = []
    for i in range(N):
      if i % 2 != 0:
        modules.append(DecoderNE(d_model, h, d_ff, dropout, compressed=compression_rate))
      else:
        modules.append(DecoderLocal(d_model, h, d_ff, dropout, split))
        
    return nn.ModuleList(modules)
