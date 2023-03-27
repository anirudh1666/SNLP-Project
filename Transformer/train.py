from Transformer import Transformer
from Batch import Batch
from NoamOpt import NoamOpt
import pickle
import torch
import torch.nn as nn 
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

    return eng_train, de_train, encoder_vocab_len, decoder_vocab_len

def data_iterator(batch_size, X, y):
    for i in range(len(X) // batch_size):
        X_batch = X[i * batch_size: (i + 1) * batch_size, :]
        y_batch = y[i * batch_size: (i + 1) * batch_size, :]

        yield Batch(X_batch, y_batch, 0)

if __name__ == '__main__':
    data = pickle.load(open('english-german-both.pkl', 'rb'))
    X, y, src_vocab_len, tgt_vocab_len = preprocess(data)

    transformer = Transformer(src_vocab_len, tgt_vocab_len)
    criterion = nn.CrossEntropyLoss()
    optimiser = NoamOpt(512, 2, 4000, torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    BATCH_SIZE = 50 # 1000 % 50 == 0 so batch size evenly divides total training set size
    EPOCHS = 10
    iterator = data_iterator(BATCH_SIZE, X, y)
    for epoch in range(EPOCHS):
        transformer.train()

        for i, batch in enumerate(iterator):
            out = transformer(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss = criterion(out.contiguous().view(-1, out.size(-1)), batch.tgt_y.contiguous().view(-1)) / batch.ntokens
            loss.backward()
            optimiser.step()
            optimiser.optimizer.zero_grad()
            
            if i % 100 == 0:
                print(loss.item() * batch.ntokens)

        transformer.eval()




