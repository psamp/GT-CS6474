import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from nltk import FreqDist
from sklearn.svm import SVC
from classifier import run_classifier

# nltk.download('stopwords')
stop = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')


def n_grams(X_train, X_test, y_train, y_true):

    def feature(data):
        tokenized_request_text_set = tokenize(data)
        unigrams, bigrams = find_top_500_ngrams(tokenized_request_text_set)
        ngram_feature = calc_features(
            unigrams, bigrams, tokenized_request_text_set)
        return ngram_feature

    train_ngram_feature = feature(X_train)
    test_ngram_feature = feature(X_test)
    c = SVC(kernel='linear', probability=True)
    
    run_classifier(c, train_ngram_feature, test_ngram_feature, y_train, y_true)

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
    list_unigrams = list(
        map(lambda ngram_freq: ngram_freq[0][0], list(unigrams)))
    list_bigrams = list(
        map(lambda ngram_freq: ngram_freq[0], list(bigrams)))

    # return ngrams
    return list_unigrams, list_bigrams


def calc_features(unigrams, bigrams, tokens):
    features = []
    for t in tokens:
        token_set = t.split()
        ngram_vector = []

        for u in unigrams:
            ngram_vector.append(1 if u in token_set else 0)

        for bigram in bigrams:
            pair = ()
            for i in range(1, len(token_set)):
                pair = (token_set[i-1], token_set[i])
                if pair == bigram:
                    break
            ngram_vector.append(1 if pair == bigram else 0)

        features.append(ngram_vector)
    return features
