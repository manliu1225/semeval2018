from collections import defaultdict
import logging
import re
import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from stemming.porter2 import stem as porter_stem
from sklearn.base import BaseEstimator, TransformerMixin
from . import BaseSentimentClassifier, PureTransformer, ListCountVectorizer, BasicTokenizer, NGramTransformer, CharNGramTransformer, CMUArkTweetPOSTagger, CMUArkTweetBrownClusters, LowercaseTransformer, Negater
from .lexicons import NRCEmotionLexicon, NRCHashtagEmotionLexicon, MaxDiffTwitterLexicon, NRCHashtagSentimentWithContextUnigrams, NRCHashtagSentimentWithContextBigrams, NRCHashtagSentimentLexiconUnigrams, NRCHashtagSentimentLexiconBigrams, Sentiment140WithContextUnigrams, Sentiment140WithContextBigrams, Sentiment140LexiconUnigrams, Sentiment140LexiconBigrams, YelpReviewsLexiconUnigrams, YelpReviewsLexiconBigrams, AmazonLaptopsReviewsLexiconUnigrams, AmazonLaptopsReviewsLexiconBigrams, MPQAEffectLexicon, MPQASubjectivityLexicon, HarvardInquirerLexicon, BingLiuLexicon, AFINN111Lexicon, SentiWordNetLexicon, LoughranMcDonaldLexicon, SentimentSpecificWordEmbeddings

from sklearn.base import BaseEstimator, TransformerMixin
logger = logging.getLogger(__name__)

'''liumandeMacBook-Air:twitter_emoij_prediction liuman$ python test.py --X_train /Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_train_1.text --y_train /Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_train_1.labels --X_test /Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_test_1.text --y_test /Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_test_1.labels '''
class Print(BaseEstimator, TransformerMixin):
  def fit(self, x, y):
    print x.shape

class GBDTSentimentClassifier(BaseSentimentClassifier):
  """
  Implementation of NRC by Saif M. Mohammad, Svetlana Kiritchenko, and Xiaodan Zhu. 2013. [NRC-Canada: Building the State-of-the-Art in Sentiment Analysis of Tweets](http://www.aclweb.org/anthology/S13-2053). In _Proceedings of the 7th International Workshop on Semantic Evaluation (SemEval 2013)_, pages 321-327.
  """


#   def __init__(self, **kwargs):
#     super(GBDTSentimentClassifier, self).__init__(classifier=lgb.LGBMClassifier(
#     objective='multiclass',
#     num_class=20,
#     # max_depth=6,
#     # num_leaves=64,
#     silent=False,
#     learning_rate=0.1,
#     # n_estimators=700,
#     nthread=16,
#     is_unbalance=True

# ), **kwargs)
  def __init__(self, **kwargs):
    super(GBDTSentimentClassifier, self).__init__(classifier=lgb.LGBMClassifier(
    objective='multiclass',
    num_class=20,
    # max_depth=6,
    # num_leaves=64,
    silent=False,
    learning_rate=0.1,
    # n_estimators=700,
    nthread=16,
    is_unbalance=True,
    ))
    self.set_params(**kwargs)
  def _make_preprocessor(self):
    def lexicon_pipeline(lexicon): return make_pipeline(LexiconFeatures(lexicon), DictVectorizer())

    unigram_lexicons_features = make_union(
      lexicon_pipeline(NRCEmotionLexicon()),
      lexicon_pipeline(NRCHashtagEmotionLexicon()),
      lexicon_pipeline(MaxDiffTwitterLexicon()),
      lexicon_pipeline(NRCHashtagSentimentWithContextUnigrams()),
      lexicon_pipeline(NRCHashtagSentimentLexiconUnigrams()),
      lexicon_pipeline(Sentiment140WithContextUnigrams()),
      lexicon_pipeline(Sentiment140LexiconUnigrams()),
      lexicon_pipeline(YelpReviewsLexiconUnigrams()),
      lexicon_pipeline(AmazonLaptopsReviewsLexiconUnigrams()),
      lexicon_pipeline(MPQAEffectLexicon()),
      lexicon_pipeline(MPQASubjectivityLexicon()),
      lexicon_pipeline(HarvardInquirerLexicon()),
      lexicon_pipeline(BingLiuLexicon()),
      lexicon_pipeline(AFINN111Lexicon()),
      lexicon_pipeline(SentiWordNetLexicon()),
      lexicon_pipeline(LoughranMcDonaldLexicon()),
    )

    bigram_lexicons_features = make_union(
      lexicon_pipeline(NRCHashtagSentimentWithContextBigrams()),
      lexicon_pipeline(NRCHashtagSentimentLexiconBigrams()),
      lexicon_pipeline(Sentiment140WithContextBigrams()),
      lexicon_pipeline(Sentiment140LexiconBigrams()),
      lexicon_pipeline(YelpReviewsLexiconBigrams()),
      lexicon_pipeline(AmazonLaptopsReviewsLexiconBigrams()),
      lexicon_pipeline(MPQAEffectLexicon()),
    )

    preprocessor = make_pipeline(
      BasicTokenizer(),
      make_union(
        make_pipeline(CMUArkTweetPOSTagger(), ListCountVectorizer(lowercase=False, binary=True)),  # POS features
        # make_pipeline(W2Vembedding()),
        make_pipeline(CharNGramTransformer([1, 2, 3]), ListCountVectorizer(lowercase=True, max_features=10000, binary=True)),  # Character n-grams
        make_pipeline(LowercaseTransformer(), CMUArkTweetBrownClusters(), ListCountVectorizer(lowercase=False, binary=True)),  # brown clusters
        make_pipeline(
          Negater(),
          make_union(
            make_pipeline(NGramTransformer([3, 4]), ListCountVectorizer(lowercase=True,  max_features=10000, binary=True)),  # ngram word
            make_pipeline(CountingFeatures(), DictVectorizer()),  # allcaps, punctuations, lengthening, emoticons, etc. counting feature
            make_pipeline(LowercaseTransformer(), unigram_lexicons_features),  # unigram lexicon features
            ListCountVectorizer(lowercase=True, max_features=10000, binary=True),  # ngram word
            make_pipeline(
              LowercaseTransformer(),
              NGramTransformer(2),  
              make_union(
                bigram_lexicons_features,  # bigram lexicon features
                ListCountVectorizer(lowercase=True, max_features=10000, binary=True),  # ngram word
              ),
            ),
          ),
        ),
        make_pipeline(LowercaseTransformer(), SSWEFeatures()),
        make_pipeline(
          NormalizedTokens(),
          CollapsedTokens(),
          PorterStemmer(),
          Negater(),
          make_union(
            ListCountVectorizer(lowercase=False, max_features=10000, binary=True),  # processed word
            make_pipeline(ClusterFeaturesWithNegation(), ListCountVectorizer(lowercase=False, binary=True)),  # processed cluster features
        ),
        )
      ),
    )

    return preprocessor


class NormalizedTokens(PureTransformer):
  """Normalize digits and lowercase tokens."""

  re_digit = re.compile(ur'\d')

  def transform_one(self, toks):
    transformed = []
    for tok in toks:
      tok = tok.lower()
      tok = self.re_digit.sub('0', tok)

      transformed.append(tok)
    

    return transformed
  

class SSWEFeatures(PureTransformer):
  """Get the SSWE vectors for each token and summarize them using min, max, mean, abs_min, and abs_max."""

  def __init__(self, stemmed=False):
    self._sswe = SentimentSpecificWordEmbeddings(stemmed=stemmed)
  

  def transform_one(self, toks):
    dims = self._sswe.dimensions
    W = np.zeros((len(toks), dims))
    for i, tok in enumerate(toks): W[i, :] = self._sswe[tok]


    return np.concatenate((
      np.min(W, axis=0),
      np.max(W, axis=0),
      np.mean(W, axis=0),
      W[np.argmax(np.abs(W), axis=0), np.arange(0, dims)],
      W[np.argmin(np.abs(W), axis=0), np.arange(0, dims)]
    ))
  



re_contiguous_punct = re.compile(r'[\?\!][\?\!]+')
re_positive_emoticon = re.compile(r'^(\:+[\-\+]?\)+|\(+[\-\+]?\:+)$')
re_negative_emoticon = re.compile(r'^(\:+[\-\+]?\(+|\)+[\-\+]?\:+)$')


class CountingFeatures(PureTransformer):
  def transform_one(self, toks):
    hashtags = 0
    allcaps = 0
    contiguous_punct = 0
    last_punct = 'x'
    positive_emoticon, negative_emoticon = False, False
    lengthened = 0
    mentions = 0
    urls = 0

    for tok in toks:
      if tok.startswith('#'): hashtags += 1
      if tok.isupper(): allcaps += 1
      if re_contiguous_punct.match(tok): contiguous_punct += 1

      if tok.endswith('!'): last_punct = u'!'
      elif tok.endswith('?'): last_punct = u'?'

      if re_positive_emoticon.match(tok): positive_emoticon = True
      elif re_negative_emoticon.match(tok): negative_emoticon = True

      for i in xrange(len(tok) - 2):
        if tok[i].isalpha() and tok[i] == tok[i+1] and tok[i] == tok[i+2]:
          lengthened += 1
          break
        
      

      if tok == '_url_': urls += 1
      if tok == '@mention': mentions += 1
    

    return dict(hashtags=hashtags, allcaps=allcaps, contiguous_punct=contiguous_punct, last_punct=last_punct, positive_emoticon=positive_emoticon, negative_emoticon=negative_emoticon, lengthened=lengthened, urls=urls, mentions=mentions)
  



class LexiconFeatures(PureTransformer):
  def __init__(self, lexicon):
    self._lexicon = lexicon
    self._have_NEG = lexicon.have_NEG
    self._have_NEGFIRST = lexicon.have_NEGFIRST
  

  def transform_one(self, toks):
    features = defaultdict(float)
    prev_negated = False
    for i, tok in enumerate(toks):
      cur_negated = tok.endswith('_NEG')

      if cur_negated: 
        tok = tok[:-4].lower() + '_NEG'
      else: tok = tok.lower()

      if self._have_NEG:
        w, negated = tok, ''
        if self._have_NEGFIRST and not prev_negated and cur_negated: w += 'FIRST'

      elif cur_negated: w, negated = tok[:-4], '_NEG'
      else: w, negated = tok, ''
      prev_negated = cur_negated

      for p, s in self._lexicon[w].iteritems():
        featname = p + negated

        if s > 0.0:
          features[featname + '_count'] += 1
          features[featname + '_last'] = s
        
        features[featname + '_sum'] += s
        features[featname + '_max'] = max(s, features[featname + '_max'])
      
    
    return features
  




class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
      self.MAX_LENGTH = 0.0
      total_len = 0.0
      for e in X: 
        total_len += len(e)
        self.MAX_LENGTH = max(self.MAX_LENGTH, len(e))
#      print 'max length of sentences is ...'
#     print self.MAX_LENGTH
#     print 'mean length of sentences is ...'
#     print total_len/len(X)
      return self

    def transform(self, X):
      X_new = []
      for line in X:
        X_new.append([self.word2vec[word] if word in self.word2vec else np.zeros(self.dim) for word in line])
      X_new = np.array(X_new)
      X = pad_sequences(X_new, self.MAX_LENGTH)
#      print 'in MeanEmbeddingVectorizer X.shape is ...'
#     print X.shape
      return X.reshape(X.shape[0],X.shape[1]*X.shape[2])

    def get_params(self, deep=True): return dict()

class TfidfEmbeddingVectorizer(BaseEstimator, TransformerMixin):
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
#        print self.word2weight
        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
class W2Vembedding(MeanEmbeddingVectorizer):
  def __init__(self, **kwargs):
    W2V_FILE = os.path.join(os.getcwd(), "tweets/resources/tweet_glove/glove.twitter.27B.50d.txt")
    with open(W2V_FILE, "rb") as lines:
      w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
    super(W2Vembedding, self).__init__(w2v)

  def get_params(self, deep=True): return dict()

class NormalizedTokens(PureTransformer):
  """Normalize digits and lowercase tokens."""

  re_digit = re.compile(ur'\d')

  def transform_one(self, toks):
    transformed = []
    for tok in toks:
      tok = tok.lower()
      tok = self.re_digit.sub('0', tok)

      transformed.append(tok)
    

    return transformed
  



class CollapsedTokens(PureTransformer):
  """Collapse tokens with repeated letters."""

  def transform_one(self, toks):
    transformed = []
    for tok in toks:
      if len(tok) < 3:
        transformed.append(tok)
        continue
      

      newtok = [tok[0], tok[1]]
      for i in xrange(2, len(tok)):
        if tok[i] == tok[i-1] and tok[i] == tok[i-2]: continue
        newtok.append(tok[i])
      

      transformed.append(u''.join(newtok))
    

    return transformed
  



class PorterStemmer(PureTransformer):
  """Porter stem tokens."""

  def transform_one(self, toks):
    return map(porter_stem, toks)
  



class ClusterFeaturesWithNegation(PureTransformer):
  def __init__(self):
    self._clusters = CMUArkTweetBrownClusters()

  def transform_one(self, toks):
    transformed = []
    for tok in toks:
      negated = tok.endswith('_NEG')
      w = tok[:-4] if negated else tok
      cluster = self._clusters._clusters.get(w)
      if cluster is not None:
        transformed.append((str(cluster) + '_NEG') if negated else cluster)
    

    return transformed
  
