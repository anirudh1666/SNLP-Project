from models.transformer_decoder import TransformerDecoder
from torchnlp.encoders.text import DelimiterEncoder
from utils.transformer.label_smoothing import LabelSmoothing
from utils.transformer.noam_opt import NoamOpt
import torch.nn.functional as F
from summarizer import Summarizer
from utils.transformer.decoding import greedy_decode
from utils.general.extract_articles import getArticles
from utils.general.data_tools import setup_GPU, data_iterator
import numpy as np
import torch
import time
import os

def extract_summaries(articles, L=100, extractor=Summarizer()):
    extracted = [None for _ in range(len(articles))]
    for i, article in enumerate(articles):
        extracted[i] = ' '.join(extractor(article).split(' ')[:L])
    return extracted

def decoder_only_preprocess(train):
    SOS = '<s>'
    EOS = '</s>'

    max_len = 0
    for i in range(len(train)):
        train[i] = SOS + ' ' + train[i] + ' ' + EOS 
        max_len = max(max_len, len(train[i]))

    encoder = DelimiterEncoder(' ', train)
    tensors = []
    for datum in train:
        encoded = encoder.encode(datum)
        if len(encoded) < max_len:
            encoded = F.pad(encoded, (0, max_len - len(encoded)), 'constant', 0)
        tensors.append(encoded)
    tensors = torch.stack(tensors)

    del train
    return tensors, len(encoder.vocab), encoder

if __name__ == '__main__':

    N = 1
    L = 500
    EPOCHS = 25 
    BATCH_SIZE = 1

    train_src_fp = 'datasets/animal_tok_min5_L7.5k/train.raw.src'
    train_tgt_fp = 'datasets/animal_tok_min5_L7.5k/train.raw.tgt'

    articles = getArticles(train_src_fp, N=N) 
    articles_str = [' '.join(article) for article in articles] 
    extracted_articles = extract_summaries(articles_str, L=L)
    tgts = getArticles(train_tgt_fp, N=N)
    tgts_str = [' '.join(summary) for summary in tgts]

    # for decoder training, concat src sequence <SEPARATOR TOKEN> tgt sequence.
    train = np.array([t1 + ' <sep> ' + t2 for t1, t2 in zip(extracted_articles, tgts_str)])
    X, src_vocab_len, encoder= decoder_only_preprocess(train)

    abstractor = TransformerDecoder(src_vocab_len)
    criterion = LabelSmoothing(size=src_vocab_len, padding_idx=0, smoothing=0.1)
    optimiser = NoamOpt(512, 2, 4000, torch.optim.Adam(abstractor.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    test_article = X[0].unsqueeze(0)
    test_article_en = encoder.decode(X[0])
    test_padding = (X != 0).unsqueeze(0)
    test_len = len(X[0])
    start_symbol = encoder.encode('<s>')[0]
    end_symbol = encoder.encode('</s>')[0]

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
        print(f'\nEPOCH: {epoch} completed | Time: {elapsed} | Loss: {total_loss:.3f}\n')
        total_loss = 0

    gpred = greedy_decode(abstractor, test_article, test_padding, test_len, start_symbol, end_symbol)
    decoded = encoder.decode(gpred)
    abstractor.save(os.path.join(os.getcwd(), 'BERT_TD_25e_20n_500l_5b.pth'))
    print(decoded)
    print(test_article_en)