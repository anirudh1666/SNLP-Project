from Transformer import Transformer
from Batch import Batch
from NoamOpt import NoamOpt
import pickle
import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from torchnlp.encoders.text import DelimiterEncoder

def preprocess(data):
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'

    max_encoder_len = 0
    max_decoder_len = 0
    for i in range(data[:, 0].size):
        data[i, 0] = BOS_WORD + ' ' + data[i, 0] + ' ' + EOS_WORD
        data[i, 1] = BOS_WORD + ' ' + data[i, 1] + ' ' + EOS_WORD

        max_encoder_len = max(max_encoder_len, len(data[i, 0].split(' ')))
        max_decoder_len = max(max_decoder_len, len(data[i, 1].split(' ')))

    encoder = DelimiterEncoder(' ', data[:, 0])
    encoder_vocab_len = len(encoder.vocab)
    decoder = DelimiterEncoder(' ', data[:, 1])
    decoder_vocab_len = len(decoder.vocab)

    eng_tensors = []
    de_tensors = []
    for eng, de in data:
        eng_encoded = encoder.encode(eng)
        de_encoded = decoder.encode(de)
        
        if len(eng_encoded) < max_encoder_len:
            eng_encoded = torch.nn.functional.pad(eng_encoded, (0, max_encoder_len - len(eng_encoded)), "constant", 0)
        if len(de_encoded) < max_decoder_len:
            de_encoded = torch.nn.functional.pad(de_encoded, (0, max_decoder_len - len(de_encoded)), "constant", 0)

        eng_tensors.append(eng_encoded)
        de_tensors.append(de_encoded)

    eng_train = torch.stack(eng_tensors)
    de_train = torch.stack(de_tensors)

    return eng_train, de_train, encoder_vocab_len, decoder_vocab_len, encoder, decoder

def data_iterator(batch_size, X, y):
    for i in range(len(X) // batch_size):
        X_batch = X[i * batch_size: (i + 1) * batch_size, :]
        y_batch = y[i * batch_size: (i + 1) * batch_size, :]

        yield Batch(X_batch, y_batch, 0)

def subsequent_mask(size):
    shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def beam_search(model, src, src_mask, max_len, start_symbol, beam_width):
    enc_out = model.encode(src, src_mask)
    possible_ys = torch.zeros(beam_width, 1, 1).fill_(start_symbol).type_as(src.data)
    branch_probs = torch.ones(beam_width)
    for i in range(max_len - 1):
        if i == 0:
            # starting run. 
            curr_ys = possible_ys[0, :, :]
            out = model.decode(enc_out, src_mask, curr_ys, subsequent_mask(curr_ys.size(1)).type_as(src.data))
            prob = F.log_softmax(model._word_gen(out), dim=-1)
            top_k = torch.topk(prob, beam_width, dim=1)
            top_probs, top_words = top_k.values, top_k.indices
            for k in range(beam_width):
                branch_probs[k] *= top_probs[k]
                possible_ys[k, :, :] = torch.cat([curr_ys, torch.zeros(1, 1).fill_(top_words[k]).type_as(src.data)], dim=1)

        else:
            branches = []
            probs = torch.ones(beam_width * beam_width)
            prob_idx = 0
            for branch in range(beam_width):
                curr_ys = possible_ys[branch, :, :]
                out = model.decode(enc_out, src_mask, curr_ys, subsequent_mask(curr_ys.size(1)).type_as(src.data))
                prob = F.log_softmax(model._word_gen(out), dim=-1)
                top_k = torch.topk(prob, beam_width, dim=1)
                top_probs, top_words = top_k.values, top_k.indices
                for k in range(beam_width):
                    probs[prob_idx] = branch_probs[branch] * top_probs[k]
                    branches[prob_idx] = torch.cat([curr_ys, torch.zeros(1, 1).fill_(top_words[k]).type_as(src.data)], dim=1)
                    prob_idx += 1
            
            # we take the top K (beam width) branches from branches and probs
            top_probs, top_probs_idxs = torch.topk(probs, beam_width)
            for j in range(beam_width):
                possible_ys[j, :, :] = branches[top_probs_idxs[j]]
                branch_probs[j] = top_probs[j]
    
    _, top_sequence_idx = torch.max(branch_probs, dim=0)
    return possible_ys[top_sequence_idx, :, :]


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.unsqueeze(0) # 1d -> 2d but the first dim has len 1. [2] -> [[2]]
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data) # [[<s>, ich, bin]]
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = F.log_softmax(model._word_gen(out[:, -1]), dim=-1)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys

if __name__ == '__main__':
    data = pickle.load(open('english-german-both.pkl', 'rb'))
    X, y, src_vocab_len, tgt_vocab_len, encoder, decoder = preprocess(data)

    transformer = Transformer(src_vocab_len, tgt_vocab_len)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimiser = NoamOpt(512, 2, 4000, torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    BATCH_SIZE = 50 # 1000 % 50 == 0 so batch size evenly divides total training set size
    EPOCHS = 10
    test = encoder.encode("<s> i am an expert </s>").unsqueeze(0)
    test_pad = torch.ones(1, 1, 6)
    max_len = 6
    start = decoder.encode("<s>")[0]
    print('\nStarting Training')
    for epoch in range(EPOCHS):
        transformer.train()

        for i, batch in enumerate(data_iterator(BATCH_SIZE, X, y)):
            out = transformer(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss = criterion(out.contiguous().view(-1, out.size(-1)), batch.tgt_y.contiguous().view(-1)) / batch.ntokens
            loss.backward()
            optimiser.step()
            optimiser.optimizer.zero_grad()
            
            #if i % 10 == 0:
            #    print(loss.item() * batch.ntokens)

        transformer.eval()
        pred = greedy_decode(transformer, test, test_pad, max_len, start)
        print(f'EPOCH: {epoch}. I am an Expert - {decoder.decode(pred.squeeze(0))}')



