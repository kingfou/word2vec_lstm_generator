import numpy as np
import string
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.feature_extraction.text import TfidfVectorizer

WORD2VEC = None

def init_w2v():
    global WORD2VEC
    if not WORD2VEC:
        print('Importing glove.6B.100d.txt...')
        with open('glove/glove.6B.100d.txt', 'r') as lines:
            WORD2VEC = {
                line.split()[0]: np.array(list(map(float, line.split()[1:])))
                for line in lines
            }
    return WORD2VEC


class Word2VecVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, word2vec=None, sent_size=64):
        self.word2vec = word2vec or init_w2v()
        self.punct = list(set(string.punctuation))
        self.sent_size = sent_size
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.w2v_dim = len(iter(self.word2vec.values()).__next__())
        self.dim =  self.w2v_dim + len(self.punct)

    def get_weight(punct):
        return np.append(np.zeros(w2v_dim),
            [1.0 if punct=X else 0.0 for X in self.punct])

    def fit(self, X, y):
        # tfidf = TfidfVectorizer(analyzer=lambda x: x)
        # tfidf.fit(X)
        # max_idf = max(tfidf.idf_)
        # self.word2weight = defaultdict(
        #     lambda: max_idf,
        #     [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            self.transform_sent(words) for words in X
        ])

    def transform_sent(self, X):
        """
        Returns 3d array

        """
        pad_size = self.sent_size - len(X)
        if pad_size > 0:
            return np.pad([
                self.word2vec.get(w, self.get_weight(w)) 
                for w in X
            ], ((0, pad_size), (0, 0)), 'constant')
        else:
            return np.array([
                self.word2vec.get(w, self.get_weight(w)) 
                for w in X[:self.sent_size]
            ])
        """
        Code for tf-idf mean
        pad_size = self.sent_size - len(X)
        if pad_size > 0:
            return np.pad([
                np.mean(self.word2vec.get(w, [np.zeros(self.dim)]) * self.word2weight[w])
                for w in X
            ], (0, pad_size), 'constant')
        else:
            return np.array([
                np.mean(self.word2vec.get(w, [np.zeros(self.dim)]) * self.word2weight[w])
                for w in X[:self.sent_size]
            ])
        """
        """
        Code for 2d array
        pad_size = self.sent_size - len(X)
        if pad_size > 0:
            return np.pad([
                np.mean(self.word2vec.get(w, [np.zeros(self.dim)]))
                for w in X
            ], (0, pad_size), 'constant')
        else:
            return np.array([
                np.mean(self.word2vec.get(w, [np.zeros(self.dim)]))
                for w in X[:self.sent_size]
            ])
        """