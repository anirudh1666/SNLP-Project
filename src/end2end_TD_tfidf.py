from models.transformer_decoder import TransformerDecoder
from torchnlp.encoders.text import DelimiterEncoder
from utils.transformer.label_smoothing import LabelSmoothing
from utils.transformer.noam_opt import NoamOpt
import torch.nn.functional as F
from utils.transformer.decoding import greedy_decode_decoder_only
from utils.general.data_tools import setup_GPU, data_iterator, get_extracted, getArticles
import numpy as np
import torch
import time
import os

def decoder_only_preprocess(train, tgt, SOS='<s>', EOS='</s>', SEP='<sep>'):
    max_len = 0
    for i in range(len(train)):
        train[i] = SOS + ' ' + train[i] + ' ' + SEP + ' ' + tgt[i] + ' ' + EOS
        max_len = max(max_len, len(train[i].split(' ')))

    encoder = DelimiterEncoder(' ', train)
    tensors = []
    for datum in train:
        encoded = encoder.encode(datum)
        if len(encoded) < max_len:
            encoded = F.pad(encoded, (0, max_len - len(encoded)), 'constant', 0)
        tensors.append(encoded)
    tensors = torch.stack(tensors)

    return tensors, len(encoder.vocab), encoder

if __name__ == '__main__':
    N = 1000
    L = 1000
    EPOCHS = 200
    BATCH_SIZE = 10
    SOS_SYMBOL = '<s>'
    SEP_SYMBOL = '<sep>'
    EOS_SYMBOL = '</s>'

    train_src_fp = 'datasets/animal_tok_min5_L7.5k/train.raw.src'
    train_tgt_fp = 'datasets/animal_tok_min5_L7.5k/train.raw.tgt'

    extracted_articles = get_extracted(train_src_fp, N=N, L=L)
    tgts = [' '.join(summary) for summary in getArticles(train_tgt_fp, N=N)]
    X, src_vocab_len, encoder = decoder_only_preprocess(extracted_articles, tgts)

    test_article = extracted_articles[0]
    test_article_encoded = encoder.encode(test_article).unsqueeze(0)
    test_tgt = tgts[0]
    test_len = len(test_tgt)
    start_symbol = encoder.encode(SOS_SYMBOL)[0]
    end_symbol = encoder.encode(EOS_SYMBOL)[0]
    sep_symbol_enc = encoder.encode(SEP_SYMBOL)[0]

    del extracted_articles
    del tgts 

    abstractor = TransformerDecoder(src_vocab_len, N=12)
    criterion = LabelSmoothing(size=src_vocab_len, padding_idx=0, smoothing=0.1)
    optimiser = NoamOpt(512, 2, 4000, torch.optim.Adam(abstractor.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    setup_GPU(abstractor, X)

    print('\nStarting Training\n')
    for epoch in range(EPOCHS):
        abstractor.train()
        start = time.time()
        total_loss = 0
        for i, batch in enumerate(data_iterator(BATCH_SIZE, X, X)):
            out = abstractor(batch.src, batch.src_mask)
            loss = criterion(out.contiguous().view(-1, out.size(-1)), batch.tgt_y.contiguous().view(-1))
            total_loss += loss 
            loss.backward()
            optimiser.step()
            optimiser.optimizer.zero_grad()

        elapsed = time.time() - start
        print(f'\nEPOCH: {epoch} completed | Time: {elapsed:.3f} | Loss: {total_loss:.3f}\n')
        total_loss = 0
    print('\nCompleted Training\n')

    gpred = greedy_decode_decoder_only(abstractor, test_article_encoded, test_len, start_symbol, end_symbol, sep_symbol_enc)
    decoded = encoder.decode(gpred)
    #abstractor.save(os.path.join(os.getcwd(), 'BERT_TD_25e_20n_500l_5b.pth'))
    print(test_article)
    print(decoded)
    print(test_tgt)
