import pandas as pd
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from nltk import FreqDist
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

# nltk.download('stopwords')
stop = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')


def n_grams(data):

    # tokenize request text of incoming dataset and save into column
    def tokenize(data):
        # remove punctuation
        data['tokenized_request_text'] = data['request_text'].apply(
            lambda row: ' '.join(tokenizer.tokenize(row)))

        # remove stopwords & lowercase
        tokens = data['tokenized_request_text'].apply(lambda row: ' '.join(
            [word for word in row.lower().split() if word not in stop])).tolist()
        return tokens

    # find top 500 unigrams and top 500 bigrams
    def find_top_500_ngrams(tokens):
        unigrams = sum(
            list(map(lambda t: list(ngrams(t.split(), 1)), tokens)), [])
        bigrams = sum(
            list(map(lambda t: list(ngrams(t.split(), 2)), tokens)), [])

        # find 500 most freq
        unigrams = FreqDist(unigrams).most_common(500)
        bigrams = FreqDist(bigrams).most_common(500)

        # map into list
        def get_ngrams(ngram_freq): return ngram_freq[0][0]
        list_unigrams = list(
            map(lambda ngram_freq: ngram_freq[0][0], list(unigrams)))
        list_bigrams = list(
            map(lambda ngram_freq: ngram_freq[0], list(bigrams)))

        # return ngrams
        return list_unigrams, list_bigrams

    def calc_features(unigrams, bigrams, tokens):
        unigram_feature_vectors = []

        for token_set in tokens:
            feature_vector = []
            for u in unigrams:
                res = 1 if u in token_set else 0
                feature_vector.append(res)
            
            unigram_feature_vectors.append(feature_vector)

        return unigram_feature_vectors

    def model(data, ngrams):
        train, test = train_test_split(data, test_size=0.1)
        svmc = SVC(kernel='linear')
        # y is going to be they got a pizza...
        svmc.fit()

    tokens = tokenize(data)
    unigrams, bigrams = find_top_500_ngrams(tokens)
    unigram_feature, bigram_feature = calc_features(unigrams, bigrams, tokens)


def activity_reputation():
    pass


def narratives():
    pass
