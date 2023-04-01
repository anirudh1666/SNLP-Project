from models.transformer_ed import Transformer
from extractive_models.Cheating.CheatingExtractor import getCheatingSummarization
from extract_articles import getArticles
from utils.general.data_tools import preprocess, data_iterator
from utils.transformer.decoding import greedy_decode, beam_search
from utils.transformer.label_smoothing import LabelSmoothing
from utils.transformer.noam_opt import NoamOpt
import numpy as np
import torch
import os

def extract_summaries(articles, L=100, extractor=getCheatingSummarization):
    """
        Loop through articles : list of strings
            extracted text = extractor(article)
            take the first L words of the extracted
        returns a list of extracted text where they are all <= L words long
    """
    extracted = [None for _ in range(len(articles))]
    for i, article in enumerate(articles):
        extracted[i] = ' '.join(extractor(article).split(' ')[:L])
    return extracted

if __name__ == '__main__':

    # filepaths to data - varies per person
    train_src_fp = 'datasets/animal_tok_min5_L7.5k/train.raw.src'
    train_tgt_fp = 'datasets/animal_tok_min5_L7.5k/train.raw.tgt'
    val_src_fp = '../datasets/animal_tok_min5_L7.5k/valid.raw.src'
    val_tgt_fp = '../datasets/animal_tok_min5_L7.5k/valid.raw.tgt'
    test_src_fp = '../datasets/animal_tok_min5_L7.5k/test.raw.src'
    test_tgt_fp = '../datasets/animal_tok_min5_L7.5k/test.raw.src'

    N = 1 # how many articles to take in.
    L = 500 # we take the first L tokens (words) out of the extractive summariser. We can vary this parameter depending on the model.
    articles = getArticles(train_src_fp, N=N) 
    articles_str = ['\n'.join(article) for article in articles] # joins the paragraphs into articles - CHANGED FOR IDENTITY AND CHEATING, JOINING ON NEWLINE RATHER THAN SPACES
    extracted_articles = extract_summaries(articles_str, L=L)
    tgts = getArticles(train_tgt_fp, N=N)
    tgts_str = [' '.join(summary) for summary in tgts]

    train = np.array([[t1, t2] for t1, t2 in zip(extracted_articles, tgts_str)]) # each nested list [extracted text, abstracted text] N x 2
    # returns X (prepend SOS, append EOS, padding so theyre all the same len, indices)
    # y is the corresponding abstracted summary N x 1 (prepend SOS, appened EOS, padding so theyre all the same len, indices). List of integers corresponding to words 
    # src_vocab_len # of unique words in input articles
    # tgt vocab_len # of unique words in abstracted articles
    # encoder is a look-up table that maps list of integers to sequences
    # decoder is a look-up table that maps list ofintegers to sequences
    X, y, src_vocab_len, tgt_vocab_len, encoder, decoder = preprocess(train) 
    abstractor = Transformer(src_vocab_len, tgt_vocab_len)
    criterion = LabelSmoothing(size=tgt_vocab_len, padding_idx=0, smoothing=0.1)
    optimiser = NoamOpt(512, 2, 4000, torch.optim.Adam(abstractor.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    test_article = X[0].unsqueeze(0) # [ich] -> [[ich]]
    test_article_en = encoder.decode(test_article.squeeze(0))
    test_padding = (X != 0).unsqueeze(0)
    test_len = len(X[0])
    start_symbol = decoder.encode('<s>')[0]
    end_symbol = decoder.encode('</s>')[0]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    abstractor.to(device)
    X.to(device)
    y.to(device)

    # MAX ARTICLES = 200. EPOCHS = 25. BATCH SIZE = 20. L = 500. 
    BATCH_SIZE = 1   # < 32 and divide N.
    EPOCHS = 500       # how many times you want to iterate over the entire dataset
    print('\nStarting Training\n')
    for epoch in range(EPOCHS):
        abstractor.train()
        for i, batch in enumerate(data_iterator(BATCH_SIZE, X, y)):
            out = abstractor(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss = criterion(out.contiguous().view(-1, out.size(-1)), batch.tgt_y.contiguous().view(-1))
            loss.backward()
            optimiser.step()
            optimiser.optimizer.zero_grad()
        
        print(f'EPOCH: {epoch} completed')

    gpred = greedy_decode(abstractor, test_article, test_padding, test_len, start_symbol, end_symbol)
    #bpred = beam_search(abstractor, test_article, test_padding, test_len, start_symbol, end_symbol, 1, 0) # should output same as greedy
    #bpred2 = beam_search(abstractor, test_article, test_padding, test_len, start_symbol, end_symbol, 2, 0.1) 
    decoded = decoder.decode(gpred)
    print(decoded)
    print(test_article_en)
    #print(decoder.decode(bpred))
    #print(decoder.decode(bpred2))
    abstractor.save(os.path.join(os.getcwd(), 'BERT_TED.pth'))

