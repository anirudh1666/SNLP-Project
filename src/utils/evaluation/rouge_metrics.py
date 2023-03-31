from torchmetrics.text.rouge import ROUGEScore
from collections import defaultdict
import json


class Metrics:
    def __init__(self, name):
        self._name = name
        self._metrics = ['rouge1_fmeasure', 'rouge1_precision', 'rouge1_recall', 'rouge2_fmeasure', 
                         'rouge2_precision', 'rouge2_recall', 'rougeL_fmeasure', 'rougeL_precision', 
                         'rougeL_recall', 'rougeLsum_fmeasure', 'rougeLsum_precision', 'rougeLsum_recall']
        self.scores = defaultdict(int)

    def calculate_rouge_metrics(self, hypothesis_list, reference_list):
        '''
        Takes a two lists of strings: hypotheses (summarised articles), and a
        list of references (original articles)
        '''

        no_articles = len(hypothesis_list)
        rouge = ROUGEScore()
        scores = list(map(lambda x:
                    rouge(*x),
                    list(zip(hypothesis_list, reference_list))))
        
        for d in scores: # you can list as many input dicts as you want here
          for key, value in d.items():
            self.scores[key] += value/no_articles

    def calculate_perplexity(self, pred, target):
        '''
        takes as input: 
        Log probabilities assigned to each token in a sequence with shape [batch_size, seq_len, vocab_size]
        Ground truth values with a shape [batch_size, seq_len]
        '''
        pp = torchmetrics.functional.text.perplexity(pred, target)
        self.scores["perplexity"] = pp.item()



