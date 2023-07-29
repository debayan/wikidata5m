import json,sys
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import numpy as np

def read_sentences_from_jsonlines(jsonlines_file):
    sentences = []
    entities = []
    with open(jsonlines_file) as reader:
        for line in reader:
            line = json.loads(line.strip())
            entity = line['entity']
            sentence = line['description'][0]
            entities.append(entity)
            sentences.append(sentence)
    return entities,sentences

def compute_sbert_embeddings(entities,sentences,f, batch_size=32):
    # Load the Sentence-BERT model (pre-trained on NLI and STS benchmark datasets)
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

    # Create batches of sentences
    num_sentences = len(sentences)
    num_batches = (num_sentences + batch_size - 1) // batch_size
    for i in range(num_batches):
        print(i)
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_sentences)
        batch_sentences = sentences[start_idx:end_idx]
        batch_entities = entities[start_idx:end_idx]
        batch_embeddings = sbert_model.encode(batch_sentences)
        for ent,emb in zip(batch_entities, batch_embeddings):
            reduced_precision_array = np.round(emb.astype(np.float64), decimals=2)
            f.write(json.dumps({'entity':ent, 'embedding':reduced_precision_array.tolist()})+'\n')
    return 

if __name__ == "__main__":
    jsonlines_file = "../data/unique_valid_descriptions1.jsonlines"
    print("Loading sentences")
    entities,sentences = read_sentences_from_jsonlines(jsonlines_file)
    f = open('sbert_embeddings.jsonlines','w')
    # Specify the desired batch size
    batch_size = 3200
    print("computing embeddings with batch size %d"%(batch_size))
    compute_sbert_embeddings(entities,sentences,f, batch_size=batch_size)
    f.close()

    # The 'embeddings' variable now contains the Sentence-BERT embeddings for each sentence in the file.
    # You can use these embeddings for various downstr
