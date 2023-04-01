from models.transformer_decoder import TransformerDecoder
from models.transformer_dmca import TransformerDMCA
from utils.transformer.label_smoothing import LabelSmoothing
from torchnlp.encoders.text import DelimiterEncoder
from utils.general.data_tools import preprocess, data_iterator, subsequent_mask
from utils.transformer.decoding import greedy_decode, beam_search, greedy_decode_decoder_only, greedy_decode_DMCA
from utils.transformer.noam_opt import NoamOpt
import pickle
import torch
import time
import numpy as np

def toy_preprocess(data):

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    SEP_SYMBOL = '<sep>'

    max_encoder_len = 0

    train_size = data[:, 0].size

    new_data = []

    for i in range(train_size):
        # prepend <s> and append </s> to each sequence

        new_string = BOS_WORD + ' ' + data[i, 0] +  ' ' + SEP_SYMBOL + ' ' + data[i, 1] + ' ' + EOS_WORD
        new_data.append(new_string)

        # store the maximum sequence length, so we know how mucht o pad
        max_encoder_len = max(max_encoder_len, len((new_string).split(' ')))


    data = np.array(new_data)

    encoder = DelimiterEncoder(' ', data)
    encoder_vocab_len = len(encoder.vocab) # # of unique words in input

    
    add_tensors = []
    
    for add in data:
        eng_encoded = encoder.encode(add) # convert from string to list of indices
        
        if len(eng_encoded) < max_encoder_len:
            # perform any padding to make the sequences equal length. Append 0.
            eng_encoded = torch.nn.functional.pad(eng_encoded, (0, max_encoder_len - len(eng_encoded)), "constant", 0)

        add_tensors.append(eng_encoded)

    del data
    add_train = torch.stack(add_tensors)

    return add_train, encoder_vocab_len,encoder


if __name__ == '__main__':
    
    #data = pickle.load(open('english-german-both.pkl', 'rb'))
    data = np.array([["i am an expert"], ["ich bin eine experte"]]).T
    X, src_vocab_len,encoder = toy_preprocess(data)
    
    
    transformer = TransformerDMCA(src_vocab_len, N=2, split=1, compression_rate=3)

    criterion = LabelSmoothing(size=src_vocab_len, padding_idx=0, smoothing=0.0)
    optimiser = NoamOpt(512, 2, 4000, torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    BATCH_SIZE = 1 # 1000 % 50 == 0 so batch size evenly divides total training set size
    EPOCHS = 1000
    test = encoder.encode("<s> i am an expert <sep>").unsqueeze(0)
    test_pad = torch.ones(1, 1, 6)
    max_len = 6
    SOS_SYMBOL = '<s>'
    SEP_SYMBOL = '<sep>'
    EOS_SYMBOL = '</s>'
    start_symbol = encoder.encode(SOS_SYMBOL)[0]
    end_symbol = encoder.encode(EOS_SYMBOL)[0]
    sep_symbol_enc = encoder.encode(SEP_SYMBOL)[0]
    
    print('\nStarting Training\n')
    for epoch in range(EPOCHS):
        transformer.train()
        start = time.time()
        total_loss = 0
        
        for i, batch in enumerate(data_iterator(BATCH_SIZE, X, X)):
            
            out = transformer(batch.src, batch.pad_mask)
            loss = criterion(out.contiguous().view(-1, out.size(-1)), batch.tgt_y.contiguous().view(-1))
            total_loss += loss 
            loss.backward()
            optimiser.step()
            optimiser.optimizer.zero_grad()
            

        elapsed = time.time() - start
        print(f'\nEPOCH: {epoch} completed | Time: {elapsed:.3f} | Loss: {total_loss:.3f}\n')
        total_loss = 0

        transformer.eval()
        gpred = greedy_decode_DMCA(transformer, test, max_len, start_symbol, end_symbol, sep_symbol_enc)
        decoded = encoder.decode(gpred)

        print("Translate 'I am an expert': ", decoded)

        
        
    print('\nCompleted Training\n')

