from models.Transformer import Transformer
from summarizer import Summarizer, TransformerSummarizer
from utils.general.extract_articles import getArticles
from utils.general.data_tools import preprocess, data_iterator
from utils.transformer.label_smoothing import LabelSmoothing
from utils.transformer.noam_opt import NoamOpt
import torch

if __name__ == '__main__':
    filepath = ''
    articles = getArticles(filepath)
    articles_str = [' '.join(article) for article in articles]
    X, y, src_vocab_len, tgt_vocab_len, encoder, decoder = preprocess(articles_str)

    extractor = Summarizer()
    abstractor = Transformer(src_vocab_len, tgt_vocab_len)
    criterion = LabelSmoothing(size=tgt_vocab_len, padding_idx=0, smoothing=0)
    optimiser = NoamOpt(512, 2, 4000, torch.optim.Adam(abstractor.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    BATCH_SIZE = 64
    EPOCHS = 10
    print('\nStarting Training\n')
    abstractor.train()
    for epoch in range(EPOCHS):
        for i, batch in enumerate(data_iterator(BATCH_SIZE, X, y)):
            extracted = extractor(X)
            # NEED TO FIGURE OUT HOW TO FEED X INTO BERT, CONVERT BERT OUTPUT INTO TRANSFORMER INPUT
            out = abstractor(extracted, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss = criterion(out.contiguous().view(-1, out.size(-1)), batch.tgt_y.contiguous().view(-1))
            loss.backward()
            optimiser.step()
            optimiser.optimizer.zero_grad()

        print(f'EPOCH: {epoch} completed')


