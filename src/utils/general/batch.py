import torch 
import numpy as np
from utils.general.data_tools import subsequent_mask

class Batch:
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1] # shift right
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self._make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    def _make_std_mask(self, tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask 