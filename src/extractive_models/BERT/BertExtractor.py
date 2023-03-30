from summarizer import Summarizer
"""
Uses pretrained bert extractor from https://github.com/dmmiller612/bert-extractive-summarizer.
Define the text to summarize.
Can also define number of sentences/ratio of original to extract (Optional).
Returns the summarized text
Returns:
    string: String of summary
"""
def getBertSummarization(text, num_sentences=-1, ratio=-1):

    bert = Summarizer() # Initialize bert summarizer, example of adding hidden layers : Summarizer(hidden=[-1,-2], hidden_concat=True)
    
    # Get bert's predicted summarization, using length if specified
    if num_sentences != -1:
        return bert(text, num_sentences=num_sentences) 
    if ratio != -1:
        return bert(text, ratio=ratio)
    return bert(text)
