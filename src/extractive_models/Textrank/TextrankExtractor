import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

"""
TextRank

Step 1: Tokenizes sentences
Step 2: Creates graph
Step 3: Calculates textrank scores
Step 4: Sort sentences by textrank scores
"""
def textrank(filePath, Start=0, N=1):
    with open(filePath, 'rb') as file:
        a = 0
        text = []
        while a != Start:
            file.readline()
            a += 1

        for line in file:
            text.append(line.decode('utf-8', 'ignore').encode('ascii', 'ignore').decode('ascii'))
            a += 1
            if a == N:
                break

    main = []
    for article in text:
        main.append(article.split('<EOP>'))

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(main)
    tokens = vectorizer.get_feature_names_out()

    similarity_matrix = (X * X.T).A
    graph = nx.from_numpy_array(similarity_matrix)

    scores = nx.pagerank(graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(main)), reverse=True)

    return ranked_sentences


