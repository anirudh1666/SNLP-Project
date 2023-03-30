from torchnlp.encoders.text import DelimiterEncoder
import torch 
import numpy as np

"""
    Tools that are necessary to fetch and prepare the data to go into the models.
"""

def preprocess(data):
    """
        data: List of List of strings. Each nested list contains the input sequence (str dtype) and output sequence (str dtype)
        Returns tensor of input sequences, padded to have equal length. Shape is (# of observations, Max input seq len)
                tensor of target sequences, padded to have equal length. Shape is (# of observations, max input seq len)
                # of unique words over all the input sequences
                # of unique words over all the target sequences
                encoder of type DelimiterEncoder used to encode the input sequences
                decoder of type DelimiterDecoder used to decode the output sequences

        Prepares the data to be processed by the models.
    """
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'

    max_encoder_len = 0
    max_decoder_len = 0
    for i in range(data[:, 0].size):
        # prepend <s> and append </s> to each sequence
        data[i, 0] = BOS_WORD + ' ' + data[i, 0] + ' ' + EOS_WORD
        data[i, 1] = BOS_WORD + ' ' + data[i, 1] + ' ' + EOS_WORD

        # store the maximum sequence length, so we know how mucht o pad
        max_encoder_len = max(max_encoder_len, len(data[i, 0].split(' ')))
        max_decoder_len = max(max_decoder_len, len(data[i, 1].split(' ')))

    encoder = DelimiterEncoder(' ', data[:, 0])
    encoder_vocab_len = len(encoder.vocab) # # of unique words in input
    decoder = DelimiterEncoder(' ', data[:, 1])
    decoder_vocab_len = len(decoder.vocab) # # of unique words in output

    eng_tensors = []
    de_tensors = []
    for eng, de in data:
        eng_encoded = encoder.encode(eng) # convert from string to list of indices
        de_encoded = decoder.encode(de)
        
        if len(eng_encoded) < max_encoder_len:
            # perform any padding to make the sequences equal length. Append 0.
            eng_encoded = torch.nn.functional.pad(eng_encoded, (0, max_encoder_len - len(eng_encoded)), "constant", 0)
        if len(de_encoded) < max_decoder_len:
            de_encoded = torch.nn.functional.pad(de_encoded, (0, max_decoder_len - len(de_encoded)), "constant", 0)

        eng_tensors.append(eng_encoded)
        de_tensors.append(de_encoded)

    eng_train = torch.stack(eng_tensors)
    de_train = torch.stack(de_tensors)

    return eng_train, de_train, encoder_vocab_len, decoder_vocab_len, encoder, decoder

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

def data_iterator(batch_size, X, y):
    """
        Iterator used to efficiently generate batches to train over. 

        X : torch.tensor of shape (# of observations, max input sequence len)
        y : torch.tensor of shape (# of observations, max output sequence len)
    """
    
    for i in range(len(X) // batch_size):
        X_batch = X[i * batch_size: (i + 1) * batch_size, :]
        y_batch = y[i * batch_size: (i + 1) * batch_size, :]

        yield Batch(X_batch, y_batch, 0)

def subsequent_mask(size):
    shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0