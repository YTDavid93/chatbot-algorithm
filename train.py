import json
import pandas as pd
import numpy as np
from nltk_utils import preprocess
from numpy.linalg import norm

# open and read the data from the intents.json
with open('intents.json', 'r', encoding="utf-8") as file:
    intents = json.load(file)

# preprocess the data
patterns = []
answers = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        answers.append(intent["responses"][0])

# Step 1: Preprocess patterns
doc_a = [preprocess(pattern) for pattern in patterns]

# Step 2: Split each document into words and combine them into a single list
all_words = []
for doc in doc_a:
    all_words.extend(doc.split())

# Step 3: Create a set of unique words
total_corpus = set(all_words)
len(total_corpus)

word_count_a = dict.fromkeys(total_corpus, 0)

for doc in doc_a:
    for word in doc.split():
        word_count_a[word] += 1

pd.set_option("display.max_rows", None)
freq = pd.DataFrame([word_count_a])
freq.T


# implementation of TF-IDF algorithm
# calculating TF
def tf(word_counts, document):
    tf_dict = {}
    corpus_count = len(document)
    if corpus_count == 0:
        return tf_dict
    for word, count in word_counts.items():
        tf_dict[word] = count / float(corpus_count)

    return tf_dict

tf(word_count_a, doc_a)


# calculating IDF
def idf(documents):
    idf_dict = {}
    N = len(documents)  # Total number of documents

    idf_dict = dict.fromkeys(total_corpus, 0)

    for document in documents:
        # Iterate over words in the document, assuming 'document' is a string
        for word in set(document.split()):
            if word in idf_dict:
                idf_dict[word] += 1

    for word, df in idf_dict.items():
        idf_dict[word] = np.log10((N + 1.0) / (df + 1.0))

    return idf_dict


# Assuming 'doc_a' contains the preprocessed documents as strings
idfs = idf(doc_a)


# calculating TF-IDF
def tf_idf(tf_doc, idfs):
    tfidf_dict = {}
    for word, value in tf_doc.items():
        if word in idfs:
            tfidf_dict[word] = value * idfs[word]

    return tfidf_dict


# calculating term frequency for each element
tf_a = tf(word_count_a, doc_a)

# calculating TF-IDF for each element
tf_idf_a = tf_idf(tf_a, idfs)

# return score
document_tdidf = pd.DataFrame([tf_idf_a])
document_tdidf.T


# Implementation of consine-similarities function
# custom cosine similarity function
def cosine_similarity_custom(vec1, vec2):
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm_a * norm_b)
