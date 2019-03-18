# -*- coding: utf-8 -*-
from collections import defaultdict
import logging
import re
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline, make_union
# from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from stemming.porter2 import stem as porter_stem

from . import BaseSentimentClassifier, PureTransformer, ListCountVectorizer, BasicTokenizer, CMUArkTweetBrownClusters, Negater
from .lexicons import NRCEmotionLexicon, NRCHashtagEmotionLexicon, MaxDiffTwitterLexicon, NRCHashtagSentimentWithContextUnigrams, NRCHashtagSentimentLexiconUnigrams, Sentiment140WithContextUnigrams, Sentiment140LexiconUnigrams, YelpReviewsLexiconUnigrams, AmazonLaptopsReviewsLexiconUnigrams, MPQAEffectLexicon, MPQASubjectivityLexicon, HarvardInquirerLexicon, BingLiuLexicon, AFINN111Lexicon, SentiWordNetLexicon, LoughranMcDonaldLexicon

logger = logging.getLogger(__name__)


class GUMLTLTSentimentClassifier(BaseSentimentClassifier):
  """
  Implementation of GU-MLT-LT by Tobias Günther and Lenz Furrer. 2013. [GU-MLT-LT: Sentiment Analysis of Short Messages Using Linguistic Features and Stochastic Gradient Descent](http://www.aclweb.org/anthology/S13-2054). In _Proceedings of the 7th International Workshop on Semantic Evaluation (SemEval 2013)_, pages 328-332.
  """

  def __init__(self, **kwargs):
    # super(GUMLTLTSentimentClassifier, self).__init__(classifier=LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', warm_start=False, class_weight='balanced'), **kwargs)
    super(GUMLTLTSentimentClassifier, self).__init__(classifier=GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=1000, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=1, max_leaf_nodes=None, warm_start=False, presort='auto'), **kwargs)

  

  def _make_preprocessor(self):
    def lexicon_pipeline(lexicon): return make_pipeline(LexiconFeatures(lexicon, stemmed=True), DictVectorizer())

    unigram_lexicons_features = make_union(
      lexicon_pipeline(NRCEmotionLexicon),
      lexicon_pipeline(NRCHashtagEmotionLexicon),
      lexicon_pipeline(MaxDiffTwitterLexicon),
      lexicon_pipeline(NRCHashtagSentimentWithContextUnigrams),
      lexicon_pipeline(NRCHashtagSentimentLexiconUnigrams),
      lexicon_pipeline(Sentiment140WithContextUnigrams),
      lexicon_pipeline(Sentiment140LexiconUnigrams),
      lexicon_pipeline(YelpReviewsLexiconUnigrams),
      lexicon_pipeline(AmazonLaptopsReviewsLexiconUnigrams),
      lexicon_pipeline(MPQAEffectLexicon),
      lexicon_pipeline(MPQASubjectivityLexicon),
      lexicon_pipeline(HarvardInquirerLexicon),
      lexicon_pipeline(BingLiuLexicon),
      lexicon_pipeline(AFINN111Lexicon),
      lexicon_pipeline(SentiWordNetLexicon),
      lexicon_pipeline(LoughranMcDonaldLexicon),
    )

    preprocessor = make_pipeline(
      BasicTokenizer(),
      NormalizedTokens(),
      make_union(
        make_pipeline(Negater(), ListCountVectorizer(lowercase=False, max_features=10000, binary=True)),  # negated unigrams
        make_pipeline(
          CollapsedTokens(),
          PorterStemmer(),
          Negater(),
          make_union(
            ListCountVectorizer(lowercase=False, max_features=10000, binary=True),  # collapsed unigrams
            make_pipeline(ClusterFeaturesWithNegation(), ListCountVectorizer(lowercase=False, binary=True)),  # brown cluster features
            unigram_lexicons_features,  # using stemmed version of tokens
          ),
        ),
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
  



class LexiconFeatures(PureTransformer):
  def __init__(self, lexicon_type, stemmed=False):
    self._lexicon = lexicon_type(stemmed=stemmed)
    self._have_NEG = self._lexicon.have_NEG
    self._have_NEGFIRST = self._lexicon.have_NEGFIRST
  

  def transform_one(self, toks):
    features = defaultdict(float)
    prev_negated = False
    for i, tok in enumerate(toks):
      cur_negated = tok.endswith('_NEG')

      if cur_negated: tok = tok[:-4].lower() + '_NEG'
      else: tok = tok.lower()

      if self._have_NEG:
        w, negated = tok, ''
        if self._have_NEGFIRST and not prev_negated and cur_negated: w += 'FIRST'

      elif cur_negated: w, negated = tok[:-4], '_NEG'
      else: w, negated = tok, ''

      prev_negated = cur_negated

      for p, s in self._lexicon[w].iteritems():
        featname = p + negated

        if s > 0.0: features[featname + '_pos_sum'] += s
        if s < 0.0: features[featname + '_neg_sum'] += s
        features[featname + '_sum'] += s
      
    

    return features
  

