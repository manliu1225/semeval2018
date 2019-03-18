import logging

import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from . import BaseSentimentClassifier, BasicTokenizer, PureTransformer, LowercaseTransformer, Negater
from .lexicons import NRCEmotionLexicon, NRCHashtagEmotionLexicon, MaxDiffTwitterLexicon, NRCHashtagSentimentWithContextUnigrams, NRCHashtagSentimentLexiconUnigrams, Sentiment140WithContextUnigrams, Sentiment140LexiconUnigrams, YelpReviewsLexiconUnigrams, AmazonLaptopsReviewsLexiconUnigrams, MPQAEffectLexicon, MPQASubjectivityLexicon, HarvardInquirerLexicon, BingLiuLexicon, AFINN111Lexicon, SentiWordNetLexicon, SentimentSpecificWordEmbeddings, LoughranMcDonaldLexicon
from .nrc import LexiconFeatures, CountingFeatures, W2Vembedding
logger = logging.getLogger(__name__)


class CoolSentimentClassifier(BaseSentimentClassifier):
  """
  Implementation of Coooolll by Duyu Tang, Furu Wei, Bing Qin, Ting Liu, and Ming Zhou. 2014. [Coooolll: A Deep Learning System for Twitter Sentiment Classification](http://www.aclweb.org/anthology/S14-2033). In _Proceedings of the 8th International Workshop on Semantic Evaluation (SemEval 2014)_, pages 208-212.
  """
  def __init__(self, **kwargs):
    # super(CoolSentimentClassifier, self).__init__(classifier=LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', warm_start=False), **kwargs)
    super(CoolSentimentClassifier, self).__init__(classifier=GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=1000, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=1, max_leaf_nodes=None, warm_start=False, presort='auto'), **kwargs)

  

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

    preprocessor = make_pipeline(
      BasicTokenizer(),
      make_union(
        make_pipeline(
          Negater(),
          make_union(
            make_pipeline(CountingFeatures(), DictVectorizer()),  # allcaps, punctuations, lengthening, emoticons, etc.
            unigram_lexicons_features,  # unigram lexicon features
          ),
        ),
        make_pipeline(LowercaseTransformer(), SSWEFeatures()),
      ),
    )

    return preprocessor
  



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
  

