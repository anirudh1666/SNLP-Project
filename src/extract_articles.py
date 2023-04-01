from utils.general.data_tools import getArticles
from summarizer import Summarizer

train_src_fp = 'datasets/animal_tok_min5_L7.5k/train.src'
test_src_fp = 'datasets/animal_tok_min5_L7.5k/test.src'
bert = Summarizer()

with open('ExtractedArticles_processed_256_1000.txt', 'w') as f:
    articles = getArticles(train_src_fp, Start=0, N=200) 
    articles_str = [' '.join(article) for article in articles] # joins the paragraphs into articles - CHANGED FOR IDENTITY AND CHEATING, JOINING ON NEWLINE RATHER THAN SPACES
    for i, article in enumerate(articles_str):
        print(f'{i} / {len(articles_str)}')
        extracted = bert(article)
        f.write(extracted + '<EOA>')

with open('test_ExtractedArticles_processed_256_1000.txt', 'w') as f:
    articles = getArticles(test_src_fp, Start=0, N=40) 
    articles_str = [' '.join(article) for article in articles] # joins the paragraphs into articles - CHANGED FOR IDENTITY AND CHEATING, JOINING ON NEWLINE RATHER THAN SPACES
    for i, article in enumerate(articles_str):
        print(f'{i} / {len(articles_str)}')
        extracted = bert(article)
        f.write(extracted + '<EOA>')