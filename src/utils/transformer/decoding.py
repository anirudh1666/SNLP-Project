import torch
import torch.nn.functional as F
import numpy as np
from utils.general.data_tools import subsequent_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data) # [[<s>, ich, bin]]
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        prob = F.log_softmax(model._word_gen(out[:, -1]), dim=-1)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
                [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
      
    return ys

def get_topk(model, out, beam_width):
    log_prob = F.log_softmax(model._word_gen(out[:, -1]), dim=-1)
    top_k = torch.topk(log_prob, beam_width, dim=1)
    return top_k.values, top_k.indices
  
def beam_search(model, src, src_mask, max_len, start_symbol, beam_width, length_penalty):
    score = lambda seq, log_prob, alpha: log_prob / ((5 + len(seq))**alpha / 6**alpha)
    # the ranking is done as log_prob / penalty. However, when we rank the sequences, the length is the same
    # so the penalty is constant in a given iteration of i. Thus, we do not have to apply the scoring to all, only the
    # ones with the highest log_probability

    enc_out = model.encode(src, src_mask)
    possible_ys = [torch.zeros(1, 1).fill_(start_symbol).type_as(src.data) for _ in range(beam_width)]
    branch_probs = torch.ones(beam_width)

    # initial beam search for start token
    curr_ys = possible_ys[0]
    out = model.decode(enc_out, src_mask, curr_ys[0], subsequent_mask(curr_ys.size(1)).type_as(src.data))
    top_probs, top_words = get_topk(model, out, beam_width)
    for k in range(beam_width):
        branch_probs[k] += score(curr_ys[0], top_probs[0, k], length_penalty)
        possible_ys[k] = torch.cat([curr_ys[0], torch.zeros(1, 1).fill_(top_words.data[0, k]).type_as(src.data)], dim=1)

    for i in range(1, max_len - 1):
        branches = [None for _ in range(beam_width * beam_width)]
        probs = torch.ones(beam_width * beam_width)
        prob_idx = 0
        for branch in range(beam_width):
            curr_ys = possible_ys[branch]
            out = model.decode(enc_out, src_mask, curr_ys, subsequent_mask(curr_ys.size(1)).type_as(src.data))
            top_probs, top_words = get_topk(model, out, beam_width)

            for k in range(beam_width):
                probs[prob_idx] = branch_probs[branch] + score(curr_ys, top_probs[0, k], length_penalty)
                branches[prob_idx] = torch.cat([curr_ys, torch.zeros(1, 1).fill_(top_words[0, k]).type_as(src.data)], dim=1)
                prob_idx += 1
            
        # we take the top K (beam width) branches from branches and probs
        top_probs, top_probs_idxs = torch.topk(probs, beam_width)
        for j in range(beam_width):
            possible_ys[j] = branches[top_probs_idxs[j]]
            branch_probs[j] = top_probs[j]
    
    _, top_sequence_idx = torch.max(branch_probs, dim=0)
    return possible_ys[top_sequence_idx]
  