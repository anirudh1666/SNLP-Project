from ExtractArticles import getArticles
from summarizer import Summarizer

"""
Uses pretrained bert extractor from https://github.com/dmmiller612/bert-extractive-summarizer.
Define the file path of the document you want to read through. 
Additionally define article to read, default is 0.
Can also define number of sentences/ratio of original to extract (Optional).
Returns the summarized text text and the original text.
Returns:
    (string, string): Tuple in format (summary, original)
"""
def getBertSummarization(filepath, article=0, num_sentences=-1, ratio=-1):
    articles = getArticles(filepath, article, 1) # Get 1 article from file
    article1 = ' '.join(articles[0]) # Articles in list of list, so get first one and make it a single string

    bert = Summarizer() # Initialize bert summarizer, example of adding hidden layers : Summarizer(hidden=[-1,-2], hidden_concat=True)
    
    # Get bert's predicted summarization, using length if specified
    if num_sentences != -1:
        out = bert(article1, num_sentences=num_sentences) 
    elif ratio != -1:
        out = bert(article1, ratio=ratio)
    else:
        out = bert(article1)
    
    return (out, article1) # Return (summary, original) tuple