import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

"""
TextRank

Step 1: Tokenizes sentences
Step 2: Creates graph
Step 3: Calculates textrank scores
Step 4: Sort sentences by textrank scores
"""
def textrank(text):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(text)
    tokens = vectorizer.get_feature_names_out()

    similarity_matrix = (X * X.T).A
    graph = nx.from_numpy_array(similarity_matrix)

    scores = nx.pagerank(graph)

    ranked_sentences = np.array(sorted(((scores[i], s) for i, s in enumerate(text)), reverse=True))
    
    return ' '.join(ranked_sentences[:, 1])


