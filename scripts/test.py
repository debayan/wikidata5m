import nltk
import networkx as nx
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
import string

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize the text into words
    words = word_tokenize(text.lower())

    # Remove punctuation and stopwords
    stop_words = set(stopwords.words("english") + list(string.punctuation))
    words = [word for word in words if word not in stop_words]

    # Stem the words
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return words

def build_graph(words):
    word_freq = FreqDist(words)
    co_occurrences = {}

    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            word_i = words[i]
            word_j = words[j]

            if word_i != word_j:
                co_occurrences[(word_i, word_j)] = co_occurrences.get((word_i, word_j), 0) + 1

    graph = nx.Graph()
    for (word_i, word_j), weight in co_occurrences.items():
        graph.add_edge(word_i, word_j, weight=weight)

    return graph

def text_rank(text, top_n=5):
    words = preprocess_text(text)
    graph = build_graph(words)
    scores = nx.pagerank(graph)

    top_words = sorted(((scores[word], word) for word in set(words)), reverse=True)
    top_words = [word for score, word in top_words[:top_n]]

    return top_words

# Example usage:
text_to_summarize = """
    TextRank is an extractive text summarization technique based on the PageRank algorithm used in search engines.
    It extracts the most important sentences from a given text to form a summary. TextRank works by representing
    the text as a graph, with sentences as nodes and edges based on the co-occurrence of words in sentences.
    The algorithm then calculates the importance of each sentence based on the graph structure and selects the
    top-ranked sentences as the summary.
"""

top_words = text_rank(text_to_summarize, top_n=10)
print(top_words)


