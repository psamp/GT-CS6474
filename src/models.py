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
        data['tokenized_request_text'] = data['request_text'].apply(lambda row: ' '.join(tokenizer.tokenize(row)))

        # remove stopwords & lowercase
        tokens = data['tokenized_request_text'].apply(lambda row: ' '.join([word for word in row.lower().split() if word not in stop])).tolist()
        return tokens
    
    # find unigrams and bigrams
    def find_ngrams(tokens):        
        unigrams = sum(list(map(lambda t: list(ngrams(t.split(), 1)), tokens)), [])
        bigrams = sum(list(map(lambda t: list(ngrams(t.split(), 2)), tokens)), [])

        # find most freq
        unigrams = FreqDist(unigrams).most_common(500)
        bigrams = FreqDist(bigrams).most_common(500)

        # word_vectorizer = CountVectorizer(ngram_range=(1,2), analyzer='word')
        # top_500_unigrams = word_vectorizer.fit_transform(unigrams)
        # top_500_bigrams = word_vectorizer.fit_transform(bigrams)

        # print("💗", top_500_unigrams)

        # return FreqDist(unigrams).most_common(500), FreqDist(bigrams).most_common(500)
        
        # word_vectorizer = CountVectorizer(ngram_range=(1,2), analyzer='word')
        # sparse_matrix = word_vectorizer.fit_transform(data['tokenized_request_text'])
        # frequencies = sum(sparse_matrix).toarray()[0]
        # ngrams = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['freq'])
        # ngrams.sort_values(by=['freq'])
        # print(ngrams.head(30))
    
    def model(data, ngrams):
        train, test = train_test_split(data, test_size=0.1)
        svmc = SVC(kernel='linear')
        # y is going to be they got a pizza...
        svmc.fit()

    tokens = tokenize(data)
    grams = find_ngrams(tokens)




def activity_reputation():
    pass

def narratives():
    pass

