from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.functional.text import perplexity
from collections import defaultdict
import torch


class Metrics:
    def __init__(self, name):
        self._name = name
        self.scores = defaultdict(int)
        self.scores["name"] = name

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
            self.scores[key] += value.item()/no_articles

        with open(f'{self._name}.txt', 'w') as f:
          for key, value in self.scores.items():
            f.write(key + ": " + str(value) + "\n")

    def calculate_perplexity(self, pred, target):
        '''
        takes as input: 
        Log probabilities assigned to each token in a sequence with shape [batch_size, seq_len, vocab_size]
        Ground truth values with a shape [batch_size, seq_len]
        '''
        pp = perplexity(pred, target)
        self.scores["perplexity"] = pp.item()

        with open(f'{self._name}.txt', 'w') as f:
          for key, value in self.scores.items():
            f.write(key + ": " + str(value) + "\n")



