import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
from utils.transformer.positional_embedding import PositionalEmbedding
from utils.transformer.encoder import Encoder
from utils.transformer.decoder import Decoder
from utils.general.data_tools import subsequent_mask

class Transformer(nn.Module):
  def __init__(self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    super().__init__()
    self._encoders = nn.ModuleList([Encoder(dropout, h, d_model, d_ff) for _ in range(N)])
    self._encoder_norm = nn.LayerNorm(d_model)
    self._decoders = nn.ModuleList([Decoder(d_model, h, d_ff, dropout) for _ in range(N)])
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
  
  def save(self):
    torch.save(self.state_dict())
  
  def load(self, path):
    self.load_state_dict(torch.load(path))
    self.eval()

  def greedy_decode(self, src, src_mask, max_len, start_symbol):
    memory = self.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data) # [[<s>, ich, bin]]
    for i in range(max_len - 1):
      out = self.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
      prob = F.log_softmax(self._word_gen(out[:, -1]), dim=-1)
      _, next_word = torch.max(prob, dim=1)
      next_word = next_word.data[0]
      ys = torch.cat(
          [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
      
    return ys
  
  def beam_search(self, src, src_mask, max_len, start_symbol, beam_width):
    enc_out = self.encode(src, src_mask)
    possible_ys = [torch.zeros(1, 1).fill_(start_symbol).type_as(src.data) for _ in range(beam_width)]
    branch_probs = torch.ones(beam_width)
    for i in range(max_len - 1):
        if i == 0:
            # starting run. 
            curr_ys = possible_ys[0]
            out = self.decode(enc_out, src_mask, curr_ys, subsequent_mask(curr_ys.size(1)).type_as(src.data))
            prob = F.log_softmax(self._word_gen(out[:, -1]), dim=-1)
            top_k = torch.topk(prob, beam_width, dim=1)
            top_probs, top_words = top_k.values, top_k.indices
            for k in range(beam_width):
                branch_probs[k] *= np.exp(top_probs[0, k].detach().numpy())
                possible_ys[k] = torch.cat([curr_ys, torch.zeros(1, 1).fill_(top_words.data[0, k]).type_as(src.data)], dim=1)

        else:
            branches = [None for _ in range(beam_width * beam_width)]
            probs = torch.ones(beam_width * beam_width)
            prob_idx = 0
            for branch in range(beam_width):
                curr_ys = possible_ys[branch]
                out = self.decode(enc_out, src_mask, curr_ys, subsequent_mask(curr_ys.size(1)).type_as(src.data))
                prob = F.log_softmax(self._word_gen(out[:, -1]), dim=-1)
                top_k = torch.topk(prob, beam_width, dim=1)
                top_probs, top_words = top_k.values, top_k.indices
                for k in range(beam_width):
                    probs[prob_idx] = branch_probs[branch] * np.exp(top_probs[0, k].detach().numpy())
                    branches[prob_idx] = torch.cat([curr_ys, torch.zeros(1, 1).fill_(top_words[0, k]).type_as(src.data)], dim=1)
                    prob_idx += 1
            
            # we take the top K (beam width) branches from branches and probs
            top_probs, top_probs_idxs = torch.topk(probs, beam_width)
            for j in range(beam_width):
                possible_ys[j] = branches[top_probs_idxs[j]]
                branch_probs[j] = top_probs[j]
    
    _, top_sequence_idx = torch.max(branch_probs, dim=0)
    return possible_ys[top_sequence_idx]
  