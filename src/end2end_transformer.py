from models.Transformer import Transformer
from summarizer import Summarizer, TransformerSummarizer
from utils.general.extract_articles import getArticles
from utils.general.data_tools import preprocess, data_iterator
from utils.transformer.decoding import greedy_decode
from utils.transformer.label_smoothing import LabelSmoothing
from utils.transformer.noam_opt import NoamOpt
import numpy as np
import torch
import os

def extract_summaries(articles, extractor=Summarizer()):
    extracted = [None for _ in range(len(articles))]
    for i, article in enumerate(articles):
        extracted[i] = extractor(article)
    return extracted

if __name__ == '__main__':

    train_src_fp = 'datasets/animal_tok_min5_L7.5k/train.raw.src'
    train_tgt_fp = 'datasets/animal_tok_min5_L7.5k/train.raw.tgt'
    val_src_fp = '../datasets/animal_tok_min5_L7.5k/valid.raw.src'
    val_tgt_fp = '../datasets/animal_tok_min5_L7.5k/valid.raw.tgt'
    test_src_fp = '../datasets/animal_tok_min5_L7.5k/test.raw.src'
    test_tgt_fp = '../datasets/animal_tok_min5_L7.5k/test.raw.src'

    N = 1
    articles = getArticles(train_src_fp, N=N)
    articles_str = [' '.join(article) for article in articles]
    extracted_articles = extract_summaries(articles_str)
    tgts = getArticles(train_tgt_fp, N=N)
    tgts_str = [' '.join(summary) for summary in tgts]

    train = np.array([[t1, t2] for t1, t2 in zip(extracted_articles, tgts_str)])
    X, y, src_vocab_len, tgt_vocab_len, encoder, decoder = preprocess(train)
    abstractor = Transformer(src_vocab_len, tgt_vocab_len)
    criterion = LabelSmoothing(size=tgt_vocab_len, padding_idx=0, smoothing=0)
    optimiser = NoamOpt(512, 2, 4000, torch.optim.Adam(abstractor.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    test_article = X[0].unsqueeze(0)
    test_article_en = encoder.decode(test_article.squeeze(0))
    test_padding = (X != 0).unsqueeze(0)
    test_len = len(X[0])
    start_symbol = decoder.encode('<s>')[0]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    abstractor.to(device)
    X.to(device)
    y.to(device)

    BATCH_SIZE = 1
    EPOCHS = 10
    print('\nStarting Training\n')
    for epoch in range(EPOCHS):
        abstractor.train()
        for i, batch in enumerate(data_iterator(BATCH_SIZE, X, y)):
            out = abstractor(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss = criterion(out.contiguous().view(-1, out.size(-1)), batch.tgt_y.contiguous().view(-1))
            loss.backward()
            optimiser.step()
            optimiser.optimizer.zero_grad()
        
        abstractor.eval()
        print(f'EPOCH: {epoch} completed')
        gpred = greedy_decode(abstractor, test_article, test_padding, test_len, start_symbol)
        decoded = decoder.decode(gpred)
        print(decoded)
        print(test_article_en)
        print()
        print()


