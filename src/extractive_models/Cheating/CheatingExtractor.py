import re
import numpy as np
"""
Uses pretrained bert extractor from https://github.com/dmmiller612/bert-extractive-summarizer.
Define the text to summarize.
Can also define number of sentences/ratio of original to extract (Optional).
Returns the summarized text
Returns:
    string: String of summary
"""
def getCheatingSummarization(text, num_paragraphs=5):
    paragraphs = text.split('\n')
    bigramsSets = []
    totalBigrams = set()
    for paragraph in paragraphs:
        paragraph = paragraph.lower()
        paragraph = re.sub(r'[^\w\s]', '', paragraph)
        paragraph = paragraph.replace('\n', ' ').replace('-', ' ').split(' ')
        bigramsSets.append(set([(paragraph[a], paragraph[a+1]) for a in range(len(paragraph) - 1) if (paragraph[a] != '' and paragraph[a+1] != '')]))
        totalBigrams = totalBigrams | bigramsSets[-1]
    
    scores = [len(bigramsSets[i] & totalBigrams)/len(totalBigrams) for i in range(len(paragraphs))]
    combined = [(scores[i], paragraphs[i]) for i in range(len(paragraphs))]
    topParagraphs = np.array(sorted(((v,k) for v,k in combined), reverse=True)[:num_paragraphs])

    return ' '.join(topParagraphs[:, 1])

### Example way to use this:
# from utils.general.extract_articles import getArticles
# articles = getArticles('C:/Users/Julian/Desktop/StatsNLP/dataset/company_tok_min5_L7.5k/train.raw.src', 0, 1) # Get 1 article from file
# article1 = '\n'.join(articles[0])    # Articles in list of list, so get first one and make it a single text with newline 
# print(getCheatingSummarization(article1))