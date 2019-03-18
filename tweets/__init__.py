import codecs
import json
import logging
import os
import random
import re
from warnings import warn

from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from stemming.porter2 import stem as porter_stem

from tabulate import tabulate

# from . import twokenize
from tweetokenize import Tokenizer

_basedir = os.path.dirname(os.path.abspath(__file__))
_parentdir = os.path.dirname(_basedir)
_resources_dir = os.path.join(_basedir, 'resources')

logger = logging.getLogger(__name__)

import jnius_config
jnius_config.add_options('-Xmx512m', '-XX:ParallelGCThreads=2')
# We set both CLASSPATH environment variable and jnius' internal setting because jnius is kinda cranky.
jnius_config.set_classpath(*(os.path.join(_resources_dir, jar) for jar in os.listdir(_resources_dir) if jar.endswith('.jar')))
# print(*(os.path.join(_resources_dir, jar) for jar in os.listdir(_resources_dir) if jar.endswith('.jar')))
# os.environ['CLASSPATH'] = u':'.join(jnius_config.get_classpath()) + ((':' + os.environ.get('CLASSPATH')) if os.environ.get('CLASSPATH') else '')
# os.environ['CLASSPATH'] = '/Users/liuman/Documents/semeval2018/twitter_emoij_prediction/tweets/resources/ark-tweet-nlp-0.3.2.jar'
os.environ['CLASSPATH'] = _basedir + '/resources/'

from jnius import autoclass

# LABELS_DESCRIPTION = dict(positive=1, negative=-1, neutral=0)


def load_instances(f_text, f_label):
  """
  load tweet and labels from two plain files
  :param f_text: file to read text from
  :param f_label: file to read labels from
  :returns: a list of tuples representing the instance and its label
  :rtype: list of tuples `(str, int)`
  """
  def get_data(f, label = False):
    li = []
    count = 0
    for line in f:
      if label: 
        li.append(int(line.strip()))
      else: li.append(line.strip())
      count += 1
    logger.info('Loaded {} instances from <{}>.'.format(count, f.name))
    f.close()
    return li
  texts = get_data(f_text)
  labels = get_data(f_label, label = True)
  instances = list(zip(texts, labels))
  # random.shuffle(instances)
  return instances


class BaseSentimentClassifier(Pipeline):
  """Base class subclassed by all our sentiment classifier implementation"""

  def __init__(self, classifier, **kwargs):
    super(BaseSentimentClassifier, self).__init__(steps=[('preprocessor', self._make_preprocessor()), ('classifier', classifier)])
    self.set_params(**kwargs)
    logger.debug('{} initialized.'.format(self._type_name()))
  

  def predict_all_proba(self, X):
    """
    Convenience function to return predicted probabilities but in dictionary form keyed by the classifier name.

    This is useful for ensemble classifiers to return the underlying probabilities by individual classifiers.
    """

    y_proba = self.predict_proba(X)
    classes_map = dict(zip(self.classes_, range(len(self.classes_))))
    return {self._type_name(): y_proba[:, [classes_map[-1], classes_map[0], classes_map[1]]]}

  def get_params(self):
    return self.get_params(deep = True)

  def _type_name(self): return type(self).__name__

  def _make_preprocessor(self): raise Exception('Not implemented.')



def identity(x): return x  # Because lambda functions cannot be pickled.


class ListCountVectorizer(CountVectorizer):
  """Helper class. A variant of CountVectorizer that takes a list as input."""

  def __init__(self, **kwargs):
    if 'analyzer' in kwargs: super(ListCountVectorizer, self).__init__(**kwargs)
    else: super(ListCountVectorizer, self).__init__(analyzer=identity, **kwargs)
  



class PureTransformer(TransformerMixin):
  """Helper class. A transformer that only does transformation and does not need to fit any internal parameters."""

  def fit(self, X, y=None, **fit_params): 
    return self

  def transform(self, X, **kwargs): 
    print len(X)
    print self.__class__
    return map(self.transform_one, X)

  def get_params(self, deep=True): return dict()

  def transform_one(self, x, **kwargs): return x



class BasicTokenizer(PureTransformer):
  """
  Basic tokenizer that splits a tweet using `twokenize` and replaces URL and @ mentions.

  The transformation takes as input a dictionary containing `content` and `topic`, and returns a list of normalized tokens relevant to the topic.
  To identify relevant tokens for the given topic, it looks for the first occurence of the topic word, and extract the longest contiguous string containing the topic word but not any of the punctuations/words in :attr:`re_punctuation`.

  :param ignore_topics: If `True` (default), the tokenizer will not do topic parsing.
  :type ignore_topics: bool
  """
  re_punctuation = re.compile(r'(\.|\,|\?|\!|but)')
  re_url = re.compile(r'https?\:\/\/')

  def __init__(self, **kwargs):
    super(BasicTokenizer, self).__init__()
    self.set_params(**kwargs)
  

  def transform(self, X):
    if isinstance(X, dict): warn('The input for classification, X, should be a list of dictionaries but we got {} instead.'.format(type(X).__name__), RuntimeWarning)

    return super(BasicTokenizer, self).transform(X)
  

  def transform_one(self, d):
    transformed = []
    if isinstance(d, dict): text = d['content']
    else: 
      text = d
      d = {}
    

    # toks = twokenize.tokenizeRawTweetText(text)
    gettokens = Tokenizer()
    toks = gettokens.tokenize(text)

    for tok in toks:
      if self.re_url.match(tok): transformed.append('_url_')
      elif tok.startswith('@'): transformed.append('@mention')
      else: transformed.append(tok)
    

    if not self.ignore_topics_:
      topic = d.get('topic')
      text = u' '.join(transformed)
      if topic:
        start = 0
        end = len(text)
        i = text.lower().find(topic.lower())
        if i > -1:
          matches = [m.end() for m in self.re_punctuation.finditer(text[:i])]
          if matches: start = matches[-1]
          m = self.re_punctuation.search(text[(i+len(topic)):])
          if m: end = m.start() + i + len(topic)
        

        transformed = [u'topic=' + topic] + text[start:end].split()

    return transformed
  

  def get_params(self, deep=True):
    return dict(ignore_topics=self.ignore_topics_)

  def set_params(self, ignore_topics=True):
    self.ignore_topics_ = ignore_topics
    return self
  



class LowercaseTransformer(PureTransformer):
  """Convert list of tokens to list of lowercase tokens."""

  def transform_one(self, toks):
    return map(lambda tok: tok.lower(), toks)



class NGramTransformer(PureTransformer):
  """Takes a list of tokens and returns a list of n-grams."""

  def __init__(self, n=[1]):
    self._n = n if isinstance(n, list) else [n]

  def get_params(self, deep=True):
    return dict(n=self._n)

  def transform_one(self, toks):
    L = len(toks)
    transformed = []
    for i in xrange(L):
      for n in self._n:
        if i + n > L: break
        transformed.append(u' '.join(toks[i:i+n]))
      
    

    return transformed
  



class CharNGramTransformer(PureTransformer):
  """Takes a list of tokens and returns a list of character n-grams."""

  def __init__(self, n=[1]):
    self._n = n if isinstance(n, list) else [n]
  

  def get_params(self, deep=True):
    return dict(n=self._n)

  def transform_one(self, toks):
    transformed = []
    for tok in toks:
      tok2 = u'^' + tok + u'$'

      L = len(tok2)
      for i in xrange(L):
        for n in self._n:
          if i + n > L: break
          transformed.append(u''.join(tok2[i:i+n]))
        
      
    

    return transformed
  



class Negater(PureTransformer):
  """Uses the negation rules defined by `Pang et al (2002) <http://dl.acm.org/citation.cfm?id=1118704>`_."""

  re_negation = re.compile(r'(?:^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)$)|(?:.+n\'t$)')
  re_clause_punct = re.compile(r'^[\.\:\;\!\?]+$')

  def transform_one(self, toks):
    negation = False
    transformed = []
    for tok in toks:
      if self.re_negation.match(tok):
        negation = True
        transformed.append(tok)
      elif self.re_clause_punct.match(tok):
        negation = False
        transformed.append(tok)
      elif negation: transformed.append(tok + '_NEG')
      else: transformed.append(tok)
    

    return transformed
  



class CMUArkTweetPOSTagger(PureTransformer):
  """Transformer frontend to JVM for CMU ARK POS tagger."""

  model_filename = os.path.join(_resources_dir, 'ark_tweet_nlp-20120919.model')  # do not change. problem with pickling and unpickling across filesystems?

  def __init__(self):
    self._setup()
  

  def _setup(self):
    # model_filename = self.model_filename if model_filename is None else model_filename

    Model = autoclass('cmu.arktweetnlp.impl.Model')
    self._model = Model.loadModelFromText(self.model_filename)

    FeatureExtractor = autoclass('cmu.arktweetnlp.impl.features.FeatureExtractor')
    self._featureExtractor = FeatureExtractor(self._model, False)

    self._Sentence = autoclass('cmu.arktweetnlp.impl.Sentence')
    self._ModelSentence = autoclass('cmu.arktweetnlp.impl.ModelSentence')
    logger.debug('Loaded Twitter POS tagger using model <{}>.'.format(os.path.relpath(self.model_filename, _parentdir)))

    self._initialized = True
  

  def transform_one(self, toks):
    if not getattr(self, '_initialized', False): self._setup()
    if not toks: return []

    sentence = self._Sentence()
    for tok in toks: sentence.tokens.add(tok)
    ms = self._ModelSentence(sentence.T())
    self._featureExtractor.computeFeatures(sentence, ms)
    self._model.greedyDecode(ms, False)

    tags = []
    for t in xrange(sentence.T()):
      tag = self._model.labelVocab.name(ms.labels[t])
      tags.append(tag)
    

    return tags
  

  def __getstate__(self):
    d = dict(self.__dict__)

    del d['_Sentence']
    del d['_ModelSentence']
    del d['_featureExtractor']
    del d['_model']

    return d
  

  def __setstate__(self, d):
    self._setup()

    return True
  



class CMUArkTweetBrownClusters(PureTransformer):
  """Twitter Brown clustering features."""

  filename = os.path.join(_resources_dir, 'ark_tweet_nlp-brown_clusters.txt')

  def __init__(self, filename=None, stemmed=False):
    if filename is None: filename = self.filename

    cluster_string_to_id = {}
    clusters_count = 0
    self._clusters = {}

    with codecs.open(filename, 'r', 'utf-8') as f:
      for line in f:
        try:
          c, w, _ = line.split(u'\t', 2)
          if c not in cluster_string_to_id:
            clusters_count += 1
            cluster_string_to_id[c] = clusters_count
          
          if stemmed: w = porter_stem(w)
          self._clusters[w] = cluster_string_to_id[c]
        except ValueError: pass
      
    

    logger.debug('Read {} words and {} clusters for {} (<{}>).'.format(len(self._clusters), clusters_count, self.__class__.__name__, os.path.relpath(filename, _parentdir)))
  

  def transform_one(self, toks):
    transformed = []

    for tok in toks:
      cluster = self._clusters.get(tok)
      if cluster is not None:
        transformed.append(cluster)
    

    return transformed
  

