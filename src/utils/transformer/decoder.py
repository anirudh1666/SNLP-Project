import torch 
import torch.nn as nn
from utils.transformer.multiheaded_attention import MultiHeadedAttention
from utils.transformer.mh_compressed_attention import MemoryCompressedAttention
from utils.transformer.local_attention import LocalAttention
from utils.transformer.feedforward import FeedForward

class Decoder(nn.Module):
  """
    The decoder module utilised in the Transformer architecture. We have two different versions: Multiheaded
    and Memory Compressed. 

    forward(self, x encoder_output, src_mask, tgt_mask) : torch.Tensor of shape batch size x sequence length x embedding dimension.
      Used to do a forward pass of the decoder module.
  """
  
  def __init__(self, d_model, h, d_ff, dropout, compressed=False):
    """
      d_model : int 
                Indicates the embedding dimension being used. A hyperparameter of the overall transformer.
      h       : int
                The number of heads to use in the attention function.
      d_ff    : int
                The number of nodes in the feed-forward network. Another hyperparameter of the overall transformer.
      dropout : float
                Tells us the probability of dropout occurring. 
      compressed: bool
                Indicates which attention function to use.
    """

    super().__init__()

    if not compressed:
      self._self_attn = MultiHeadedAttention(h, d_model)
      self._src_attn = MultiHeadedAttention(h, d_model)
    else:
      self._self_attn = MemoryCompressedAttention(h, d_model)
      self._src_attn = MemoryCompressedAttention(h, d_model)

    self._ff = FeedForward(d_model, d_ff, dropout)

    self._norm1 = nn.LayerNorm(d_model)
    self._norm2 = nn.LayerNorm(d_model)
    self._norm3 = nn.LayerNorm(d_model)

    self._dropout1 = nn.Dropout(dropout)
    self._dropout2 = nn.Dropout(dropout)

  def forward(self, x, encoder_output, src_mask, tgt_mask):
    """
      Performs self-attention on it's previous outputs, x. Then, perform attention with the encoder output. In 
      between, we add residual connections, dropout, and normalisation.

      Returns a tensor containing the results of the computation.

      x : torch.Tensor of shape batch size x sequence len x embed dim. 
          It is the previously output tokens of the decoder. During training, the first token is <s>
      encoder_output: torch.Tensor of shape batch size x input sequence len x embed dim
          Take the output of running the encoder over the input and perform attention with it and the decoder input, x.
      src mask: torch.Tensor of shape 1 x input sequence len
          Indicates where padding appears in the input sequence. We do not want to attend over padding tokens, so we mask them out.
      tgt_mask: torch.Tensor of shape 1 x output sequence len x output sequence len
          Upper triangular matrix of 0's at the top. Use this to prevent decoder from attending to subsequent words.
    """

    x2 = self._dropout1(self._self_attn(x, x, x, tgt_mask))
    x = self._norm1(x + x2)
    x2 = self._dropout2(self._src_attn(x, encoder_output, encoder_output, src_mask))
    x = self._norm2(x + x2)
    return self._norm3(x + self._ff(x))
  
class DecoderNE(nn.Module):
  def __init__(self, d_model, h, d_ff, dropout, compressed=False):
    super().__init__()
    if not compressed:
      self._self_attn_1 = MultiHeadedAttention(h, d_model)
      self._self_attn_2 = MultiHeadedAttention(h, d_model)
    else:
      self._self_attn_1 = MemoryCompressedAttention(h, d_model, compression_ratio=compressed)
      self._self_attn_2 = MemoryCompressedAttention(h, d_model, compression_ratio=compressed)
      
    self._ff = FeedForward(d_model, d_ff, dropout)
    self._norm1 = nn.LayerNorm(d_model)
    self._norm2 = nn.LayerNorm(d_model)
    self._norm3 = nn.LayerNorm(d_model)
    self._dropout1 = nn.Dropout(dropout)
    self._dropout2 = nn.Dropout(dropout)

  def forward(self, x, tgt_mask, src_mask=None):
    x2 = self._dropout1(self._self_attn_1(x, x, x, tgt_mask))
    x = self._norm1(x + x2)
    x2 = self._dropout2(self._self_attn_2(x, x, x, tgt_mask))  # does this need mask?
    x = self._norm2(x + x2)
    return self._norm3(x + self._ff(x))


class DecoderLocal(nn.Module):
  def __init__(self, d_model, h, d_ff, dropout, split=3):
    super().__init__()
    self._self_attn_1 = LocalAttention(h, d_model, split=split)
    self._self_attn_2 = LocalAttention(h, d_model, split=split)
      
    self._ff = FeedForward(d_model, d_ff, dropout)
    self._norm1 = nn.LayerNorm(d_model)
    self._norm2 = nn.LayerNorm(d_model)
    self._norm3 = nn.LayerNorm(d_model)
    self._dropout1 = nn.Dropout(dropout)
    self._dropout2 = nn.Dropout(dropout)

  def forward(self, x, tgt_mask):
    x2 = self._dropout1(self._self_attn_1(x, tgt_mask))
    x = self._norm1(x + x2)
    x2 = self._dropout2(self._self_attn_2(x, tgt_mask))
    x = self._norm2(x + x2)
    return self._norm3(x + self._ff(x))
