from models.MemoryCompressedTransformer import  MemoryCompressedTransformer
from models.Transformer import Transformer
from utils.transformer.label_smoothing import LabelSmoothing
from utils.general.data_tools import preprocess, data_iterator, subsequent_mask
from utils.transformer.decoding import greedy_decode, beam_search
from utils.transformer.noam_opt import NoamOpt
import pickle
import torch

if __name__ == '__main__':
    data = pickle.load(open('src/english-german-both.pkl', 'rb'))
    X, y, src_vocab_len, tgt_vocab_len, encoder, decoder = preprocess(data)

    transformer = Transformer(src_vocab_len, tgt_vocab_len, N=2)
    #transformer = MemoryCompressedTransformer(src_vocab_len, tgt_vocab_len, N=2)
    criterion = LabelSmoothing(size=tgt_vocab_len, padding_idx=0, smoothing=0.0)
    optimiser = NoamOpt(512, 2, 4000, torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    BATCH_SIZE = 20 # 1000 % 50 == 0 so batch size evenly divides total training set size
    EPOCHS = 7
    test = encoder.encode("<s> i am an expert </s>").unsqueeze(0)
    test_pad = torch.ones(1, 1, 6)
    max_len = 6
    start = decoder.encode("<s>")[0]
    print('\nStarting Training')
    
    for epoch in range(EPOCHS):
        transformer.train() 

        for i, batch in enumerate(data_iterator(BATCH_SIZE, X, y)):
            out = transformer(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss = criterion(out.contiguous().view(-1, out.size(-1)), batch.tgt_y.contiguous().view(-1))
            loss.backward()
            optimiser.step()
            optimiser.optimizer.zero_grad()
            
        transformer.eval()
        gpred = greedy_decode(transformer, test, test_pad, max_len, start)
        print([i for i in gpred])
        bpred = beam_search(transformer, test, test_pad, max_len, start, 2, 0.6)
        b1pred = beam_search(transformer, test, test_pad, max_len, start, 1, 0)
        b2pred = beam_search(transformer, test, test_pad, max_len, start, 2, 0)
        print(f'I am an Expert - Beam Search: {decoder.decode(bpred)} | Greedy Search: {decoder.decode(gpred)} | Pseudo-Greedy: {decoder.decode(b1pred)} | Beam w/ no penalty: {decoder.decode(b2pred)}')
    transformer.save()