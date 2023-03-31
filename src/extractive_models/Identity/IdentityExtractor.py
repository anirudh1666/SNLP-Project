"""
Pass in a string with newlines and number of paragraphs.
Returns first N paragraphs.
Returns: String of first N paragraphs with spaces instead of newlines
"""
def getIdentitySummarization(text, N=5):
    return ' '.join(text.split('\n')[:N])