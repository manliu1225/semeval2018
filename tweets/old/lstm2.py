from collections import defaultdict
import numpy as np
import logging
import re
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression
from keras.preprocessing.sequence import pad_sequences
from . import BaseSentimentClassifier, PureTransformer, ListCountVectorizer, BasicTokenizer, NGramTransformer, CharNGramTransformer, CMUArkTweetPOSTagger, CMUArkTweetBrownClusters, LowercaseTransformer, Negater
from .lexicons import NRCEmotionLexicon, NRCHashtagEmotionLexicon, MaxDiffTwitterLexicon, NRCHashtagSentimentWithContextUnigrams, NRCHashtagSentimentWithContextBigrams, NRCHashtagSentimentLexiconUnigrams, NRCHashtagSentimentLexiconBigrams, Sentiment140WithContextUnigrams, Sentiment140WithContextBigrams, Sentiment140LexiconUnigrams, Sentiment140LexiconBigrams, YelpReviewsLexiconUnigrams, YelpReviewsLexiconBigrams, AmazonLaptopsReviewsLexiconUnigrams, AmazonLaptopsReviewsLexiconBigrams, MPQAEffectLexicon, MPQASubjectivityLexicon, HarvardInquirerLexicon, BingLiuLexicon, AFINN111Lexicon, SentiWordNetLexicon, LoughranMcDonaldLexicon
from sklearn.base import BaseEstimator, TransformerMixin
from .dnn.bilstm import BiLSTMClassifier
logger = logging.getLogger(__name__)
W2V_FILE = os.path.join(os.getcwd(), "tweets/resources/tweet_glove/glove.twitter.27B.50d.txt")
'''liumandeMacBook-Air:twitter_emoij_prediction liuman$ python test.py --X_train /Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_train_1.text --y_train /Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_train_1.labels --X_test /Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_test_1.text --y_test /Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_test_1.labels '''
class Print(BaseEstimator, TransformerMixin):
  def fit(self, x, y):
    print x.shape

class BiLSTMSentimentClassifier(BaseSentimentClassifier):

  def __init__(self, **kwargs):
    # super(NRCSentimentClassifier, self).__init__(classifier=LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', warm_start=False), **kwargs)
    super(BiLSTMSentimentClassifier, self).__init__(classifier=BiLSTMClassifier(), **kwargs)
  #end def

  def _make_preprocessor(self):
    with open(W2V_FILE, "rb") as lines:
      w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
    preprocessor = make_pipeline(
      BasicTokenizer(), 
      MeanEmbeddingVectorizer(w2v),
      )

    return preprocessor

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
      self.MAX_LENGTH = 0.0
      total_len = 0.0
      for e in X: 
        total_len += len(e)
        self.MAX_LENGTH = max(self.MAX_LENGTH, len(e))
      print 'max length of sentences is ...'
      print self.MAX_LENGTH
      print 'mean length of sentences is ...'
      print total_len/len(X)
      return self

    def transform(self, X):
      X_new = []
      for line in X:
        X_new.append([self.word2vec[word] if word in self.word2vec else np.zeros(self.dim) for word in line])
      X_new = np.array(X_new)
      X = pad_sequences(X_new, maxlen=self.MAX_LENGTH)
      print 'in MeanEmbeddingVectorizer X.shape is ...'
      print X.shape
      return X

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        print self.word2weight
        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

