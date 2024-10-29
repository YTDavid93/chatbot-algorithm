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


class TFIDFVectorizer:
    def __init__(self):
        self.doc_a = []
        self.total_corpus = set()
        self.idf_dict = {}
        self.document_vectors = []

    def build_corpus(self, patterns):
        """Builds the document corpus and total corpus set from the provided patterns."""
        self.doc_a = [preprocess(pattern) for pattern in patterns]

        all_words = []
        for doc in self.doc_a:
            all_words.extend(doc.split())

        self.total_corpus = set(all_words)
        return self.total_corpus, self.doc_a

    def _calculate_word_counts(self, documents):
        """Calculates word counts for the provided documents."""
        word_count = dict.fromkeys(self.total_corpus, 0)

        for doc in documents:
            for word in doc.split():
                word_count[word] += 1

        return word_count

    def calculate_tf(self, word_counts, document):
        """Calculates TF for a given word count and document."""
        tf_dict = {}
        corpus_count = len(document)
        if corpus_count == 0:
            return tf_dict
        for word, count in word_counts.items():
            tf_dict[word] = count / float(corpus_count)

        return tf_dict

    def calculate_idf(self, documents):
        """Calculates Inverse Document Frequency (IDF) for the corpus."""
        N = len(documents)  # Total number of documents
        idf_dict = dict.fromkeys(self.total_corpus, 0)

        for document in documents:
            # Iterate over words in the document, assuming 'document' is a string
            for word in set(document.split()):
                if word in idf_dict:
                    idf_dict[word] += 1

            for word, df in idf_dict.items():
                idf_dict[word] = np.log10((N + 1.0) / (df + 1.0))

        self.idf_dict = idf_dict
        return idf_dict


    def calculate_tf_idf(self, tf_doc):
        """Calculates TF-IDF for the given TF dictionary using stored IDF values."""
        tfidf_dict = {}
        for word, value in tf_doc.items():
            if word in self.idf_dict:
                tfidf_dict[word] = value * self.idf_dict[word]

        return tfidf_dict

    def cosine_similarity_custom(self, vec1, vec2):
        """Calculates cosine similarity between two vectors."""
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm_a * norm_b)

    def precompute_document_vectors(self):
        """Precomputes TF-IDF vectors for all documents in the corpus."""
        for pattern in self.doc_a:
            word_count_doc = self._calculate_word_counts([pattern])
            tf_doc = self.calculate_tf(word_count_doc, pattern.split())
            tfidf_doc = self.calculate_tf_idf(tf_doc)
            vector = np.array([tfidf_doc.get(word, 0) for word in self.total_corpus])
            self.document_vectors.append(vector)

    def get_response(self, text):
        if not text:
            return  "Invalid Input"
        """Generates a response based on the input text."""
        processed_text = preprocess(text)
        print("processed_text:", processed_text)

        # Calculate word count for the input text
        input_word_count = dict.fromkeys(self.total_corpus, 0)
        for word in processed_text.split():
            if word in input_word_count:
                input_word_count[word] += 1

        # Calculate TF-IDF for the input text
        input_tf = self.calculate_tf(input_word_count, processed_text.split())
        input_tfidf = self.calculate_tf_idf(input_tf)

        # Convert input TF-IDF to vector
        input_vector = np.array([input_tfidf.get(word, 0) for word in self.total_corpus])
        print("input_vector:", input_vector)

        # Calculate cosine similarity between input vector and document vectors
        similarities = [self.cosine_similarity_custom(input_vector, doc_vector) for doc_vector in self.document_vectors]
        print("similarities:", similarities)

        # Determine the closest document
        max_similarity = np.max(similarities)

        if max_similarity > 0.4:
            best_match_index = np.argmax(similarities)
            return answers[best_match_index]
        else:
            return "I can't answer this question"

