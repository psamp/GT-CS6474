import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.svm import SVC
from classifier import run_classifier


def narratives(X_train, X_test, y_train, y_true):
    def feature(data):
        data = data['request_text']
        return calc_features(data)

    train_narr_feature = feature(X_train)
    test_narr_feature = feature(X_test)
    c = SVC(kernel='linear', probability=True)

    run_classifier(c, train_narr_feature, test_narr_feature, y_train, y_true)


def calc_features(data):
    def get_narratives():
        files = ["desire", "family", "job", "money", "student"]
        narrs = {}

        for f in files:
            words = fetch("./data/narratives/" + f + ".txt")
            narrs[f] = words

        return narrs

    narrs = get_narratives()
    narrative_features = []

    for row in data:
        row_vector = []
        white_spaced_words = len(row.split())

        if white_spaced_words:
            for narr in narrs:
                score = 0

                for word in narr:
                    score =+ row.count(word)
                row_vector.append(score/white_spaced_words)
        else:
            row_vector = [0] * 5
        
        narrative_features.append(row_vector)

    return narrative_features

def fetch(path):
    return open(path,'r').read().split('\n')
