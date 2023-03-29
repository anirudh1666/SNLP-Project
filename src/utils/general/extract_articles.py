"""
Starting from the specified article, reads N articles. 
Then splits each articles into paragraphs.
Define the file path of the document you want to read through.
Returns:
    List<List<String>>: Returns a List of all articles, each article returns a list of paragraphs.
"""
def getArticles(filePath, Start=0, N=1):
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
    
    return main
