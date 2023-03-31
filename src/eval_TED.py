"""
    Use this to evaluate TED abstractor model. 
"""
from models.transformer_ed import Transformer
from utils.general.data_tools import getArticles, preprocess, setup_GPU, data_iterator
from utils.transformer.decoding import greedy_decode
from utils.evaluation.rouge_metrics import Metrics
import numpy as np

if __name__ == '__main__':
    MODEL_PATH = '.pth'
    SOS_SYMBOL = '<s>'
    EOS_SYMBOL = '</s>'
    N = 200 # test set is 20% of train set => 200 articles
    METRIC_PATH = ''

    test_src_fp = 'datasets/animal_tok_min5_L7.5k/test.raw.src' # TODO: change this
    test_tgt_fp = 'datasets/animal_tok_min5_L7.5k/test.raw.tgt'

    # using pre-extracted text? Yes, so no need ot perform extraction
    extracted_articles = None # TODO: load in extracted articles
    tgts = [' '.join(summary) for summary in getArticles(test_tgt_fp, N=N)]
    train = np.array([[t1, t2] for t1, t2 in zip(extracted_articles, tgts)])
    X, y, src_vocab_len, tgt_vocab_len, encoder, decoder = preprocess(train)

    del extracted_articles
    del tgts
    del train 

    abstractor = Transformer(src_vocab_len, tgt_vocab_len)
    abstractor.load(MODEL_PATH)

    setup_GPU(abstractor, X, y)

    print('\nStarting Evaluation\n')
    metrics = Metrics(MODEL_PATH)
    for i, batch in enumerate(data_iterator(1, X, y)):
        print(f'{i} / {N}')

        out = greedy_decode(abstractor, batch.src, batch.src_mask, len(batch.src[0]), start_symbol, end_symbol)
        metrics.calculate_perplexity(out, batch.tgt)
        metrics.calculate_rouge_metrics(out, batch.tgt)
    
    with open(METRIC_PATH, 'w') as f:
        f.write()


