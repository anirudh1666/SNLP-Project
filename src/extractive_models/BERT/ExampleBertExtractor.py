from ExtractArticles import getArticles
from summarizer import Summarizer, TransformerSummarizer

filepath = 'dataset/film_tok_min5_L7.5k/test.raw.src' #Change depending on what articles, should always be a .raw.src file

articles = getArticles(filepath) # Get 1 article from file
article1 = ' '.join(articles[0]) # Articles in list of list, so get first one and make it a single string

bert = Summarizer()  # Initialize bert summarizer, example of adding hidden layers : Summarizer(hidden=[-1,-2], hidden_concat=True)
transformer = TransformerSummarizer() # Initialized pretrained transformer summarizer

out = bert(article1) # Get bert's predicted summarization, can specify length of summary with num_sentences=5 or ratio=0.3
out2 = transformer(article1) # Get transformer's predicted summarization, can specify lenght here also

with open('./output.txt', 'w') as file: # Write the two summarized versions of the article to a file
    file.writelines(out)
    file.writelines('\n')
    file.writelines(out2)

with open('original.txt', 'w') as f: # Write the original article to a file
    f.writelines(article1)