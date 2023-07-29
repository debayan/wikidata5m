import json,sys
import numpy as np
import re
import nltk
import networkx as nx
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
import string
import multiprocessing

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
allwords = set(nltk.corpus.words.words())

def replace_non_alpha_with_space(text):
    # Define the regular expression pattern to match non-alphanumeric characters
    pattern = r'[^a-zA-Z\s]+'
    # Use re.sub() to replace all matches with a space
    cleaned_text = re.sub(pattern, ' ', text)
    return cleaned_text


def preprocess_text(label, text):
    # Tokenize the text into words
    text = text.lower()
    label_words = label.lower().split(' ')
    text = ' '.join([word for word in text.split(' ') if word not in label_words])
    text = replace_non_alpha_with_space(text)
    text = ' '.join([word for word in text.split(' ') if word in allwords])
    words = word_tokenize(text)

    # Remove punctuation and stopwords
    stop_words = set(stopwords.words("english") + list(string.punctuation))
    words = [word for word in words if word not in stop_words]

    # Stem the words
    #stemmer = PorterStemmer()
    #words = [stemmer.stem(word) for word in words]

    return words


def build_graph(words):
    words = [word for word in words if word in allwords]
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

def text_rank(label, text, top_n=5):
    words = preprocess_text(label, text)
    graph = build_graph(words)
    scores = nx.pagerank(graph)
    top_words = sorted(((scores[word], word) for word in set(words)), reverse=True)
    top_words = [word for score, word in top_words[:top_n]]

    return top_words

def createids(entities, labels, descriptions, thread=1):
    textdict = {}
    masterdict = {}
    count = 0
    for entity,label,description in zip(entities,labels,descriptions):
        count += 1
        try:
            words = text_rank(label, description,top_n=3)
        except Exception as err:
            print(err)
            continue
        k = ':'.join(words)
        if k in textdict:
            textdict[k] += 1
        else:
            textdict[k] = 1
        masterdict[entity] = {'newid':k, 'count':textdict[k],'label':label, 'description':description}
        if count%1000 == 0:
            print(count)
            print(masterdict[entity])
    f = open('newids_%s.json'%(str(thread)),'w')
    f.write(json.dumps(masterdict,indent=4))
    f.close()

def read_sentences_from_jsonlines(jsonlines_file):
    sentences = []
    entities = []
    labels = []
    with open(jsonlines_file) as reader:
        for line in reader:
            line = json.loads(line.strip())
            entity = line['entity']
            label = line['label']
            sentence = line['description']
            labels.append(label)
            entities.append(entity)
            sentences.append(sentence)
    return entities,labels,sentences


if __name__ == "__main__":
    jsonlines_file = "../data/unique_valid_descriptions1.jsonlines"
    print("Loading sentences")
    entities,labels,sentences = read_sentences_from_jsonlines(jsonlines_file)
    l = len(entities)
    workers = 20
    args = []
    for i in range(workers):
        start = i * int(l/workers)
        end = (i+1) * int(l/workers)
        args.append((entities[start:end],labels[start:end],sentences[start:end],i))
    with multiprocessing.Pool() as pool:
        # Map the function to different arguments in parallel
        pool.starmap(createids, args)
