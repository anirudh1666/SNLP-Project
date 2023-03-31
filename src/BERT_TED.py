from models.transformer_ed import Transformer
from utils.general.data_tools import preprocess, data_iterator, setup_GPU, get_extracted, getArticles
from utils.transformer.decoding import greedy_decode, beam_search
from utils.transformer.label_smoothing import LabelSmoothing
from utils.transformer.noam_opt import NoamOpt
import numpy as np
import torch
import os
import time

if __name__ == '__main__':
    N = 1000 # how many articles to take in.
    L = 500 # we take the first L tokens (words) out of the extractive summariser. We can vary this parameter depending on the model.
    EPOCHS = 200
    BATCH_SIZE = 10
    SOS_SYMBOL = '<s>'
    EOS_SYMBOL = '</s>'

    # filepaths to data - varies per person
    train_src_fp = 'datasets/animal_tok_min5_L7.5k/train.raw.src'
    train_tgt_fp = 'datasets/animal_tok_min5_L7.5k/train.raw.tgt'

    extracted_articles = get_extracted(train_src_fp, N=N, L=L)
    tgts = [' '.join(summary) for summary in getArticles(train_tgt_fp, N=N)]
    train = np.array([[t1, t2] for t1, t2 in zip(extracted_articles, tgts)]) # each nested list [extracted text, abstracted text] N x 2

    '''
        To feed train into preprocess, train should be in the following format:
        [[extracted article 1, target article 1], ..., [extracted article 1000, target article 1000]] 
        Make sure to wrap it with a numpy array
    '''
    
    # returns X (prepend SOS, append EOS, padding so theyre all the same len, indices)
    # y is the corresponding abstracted summary N x 1 (prepend SOS, appened EOS, padding so theyre all the same len, indices). List of integers corresponding to words 
    # src_vocab_len # of unique words in input articles
    # tgt vocab_len # of unique words in abstracted articles
    # encoder is a look-up table that maps list of integers to sequences
    # decoder is a look-up table that maps list ofintegers to sequences
    X, y, src_vocab_len, tgt_vocab_len, encoder, decoder = preprocess(train) 
    test_article = X[0].unsqueeze(0)
    test_article_en = encoder.decode(test_article.squeeze(0))
    test_padding = (test_article != 0).unsqueeze(0)
    test_len = len(test_article)
    start_symbol = decoder.encode(SOS_SYMBOL)[0]
    end_symbol = decoder.encode(EOS_SYMBOL)[0]

    del extracted_articles
    del tgts
    del train

    abstractor = Transformer(src_vocab_len, tgt_vocab_len)
    criterion = LabelSmoothing(size=tgt_vocab_len, padding_idx=0, smoothing=0.1)
    optimiser = NoamOpt(512, 2, 4000, torch.optim.Adam(abstractor.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    setup_GPU(abstractor, X, y)

    print('\nStarting Training\n')
    for epoch in range(EPOCHS):
        abstractor.train()
        start = time.time()
        total_loss = 0
        for i, batch in enumerate(data_iterator(BATCH_SIZE, X, y)):
            # 20 articles, batch size = 5, i = 0..3
            print(f'Batch {i} / 3 completed')

            out = abstractor(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss = criterion(out.contiguous().view(-1, out.size(-1)), batch.tgt_y.contiguous().view(-1))
            total_loss += loss
            loss.backward()
            optimiser.step()
            optimiser.optimizer.zero_grad()
        
        elapsed = time.time() - start
        print(f'\nEPOCH: {epoch} completed | Time: {elapsed} | Loss: {total_loss:.3f}\n')
        total_loss = 0

    abstractor.save(os.path.join(os.getcwd(), 'BERT_TED_200e_1000n_500l_10b.pth')) # RENAME THIS TO FILE
    gpred = greedy_decode(abstractor, test_article, test_padding, test_len, start_symbol, end_symbol)
    decoded = decoder.decode(gpred)
    print(decoded)
    print(test_article_en)
    print(decoder.decode(y[0]))

