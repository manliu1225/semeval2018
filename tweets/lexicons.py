import codecs
import csv
import logging
import os
import re

import numpy as np

from stemming.porter2 import stem as porter_stem

logger = logging.getLogger(__name__)

_basedir = os.path.dirname(os.path.abspath(__file__))
_parentdir = os.path.dirname(_basedir)
_lexicon_data_dir = os.path.join(_basedir, 'resources', 'lexicons')


class Lexicon(object):
  filename = None
  have_NEG, have_NEGFIRST = False, False  
  def __init__(self, *args, **kwargs):
    logger.debug(self.load_lexicon(*args, **kwargs))



class AssociationScoreLexicon(Lexicon):
  """Lexicons that come with "scores" and categories."""

  def load_lexicon(self, filename=None, stemmed=False):
    if filename is None: filename = self.filename

    self._scores = {}

    with codecs.open(filename, 'r', 'utf-8') as f:
      for line in f:
        w, c, s = self._parse_line(line)
        s = float(s)
        if s == 0.0: continue
        if stemmed: w = porter_stem(w)

        if w not in self._scores: self._scores[w] = {}
        self._scores[w][c] = s
      
    

    return u'Read {} items from {} (<{}>).'.format(len(self._scores), self.__class__.__name__, os.path.relpath(filename, _parentdir))
  

  def __getitem__(self, w): return self._scores.get(w, {})

  def _parse_line(self, line): raise Exception('Not implemented.')



class NRCEmotionLexicon(AssociationScoreLexicon):
  filename = os.path.join(_lexicon_data_dir, 'nrc_emotion_lexicon-0.92.txt')

  def _parse_line(self, line):
    w, c, s = line.strip().split(u'\t', 2)
    return (w, c, s)
  



class NRCHashtagEmotionLexicon(AssociationScoreLexicon):
  filename = os.path.join(_lexicon_data_dir, 'nrc_hashtag_emotion_lexicon-0.2.txt')

  def _parse_line(self, line):
    c, w, s = line.strip().split(u'\t', 2)
    return (w, c, s)
  



class MaxDiffTwitterLexicon(AssociationScoreLexicon):
  filename = os.path.join(_lexicon_data_dir, 'maxdiff_twitter_lexicon.txt')

  def _parse_line(self, line):
    w, s = line.strip().split(u'\t', 1)
    return (w, '+ve', s)
  



class NRCHashtagSentimentLexiconUnigrams(AssociationScoreLexicon):
  filename = os.path.join(_lexicon_data_dir, 'nrc_hashtag_sentiment_lexicon-unigrams.txt')

  def _parse_line(self, line):
    w, s, _ = line.strip().split(u'\t', 2)
    return (w, '+ve', s)
  



class NRCHashtagSentimentLexiconBigrams(NRCHashtagSentimentLexiconUnigrams):
  filename = os.path.join(_lexicon_data_dir, 'nrc_hashtag_sentiment_lexicon-bigrams.txt')



class NRCHashtagSentimentWithContextUnigrams(NRCHashtagSentimentLexiconUnigrams):
  filename = os.path.join(_lexicon_data_dir, 'nrc_hashtag_sentiment_lexicon_context-unigrams.txt')
  have_NEG, have_NEGFIRST = True, True



class NRCHashtagSentimentWithContextBigrams(NRCHashtagSentimentLexiconUnigrams):
  filename = os.path.join(_lexicon_data_dir, 'nrc_hashtag_sentiment_lexicon_context-bigrams.txt')
  have_NEG, have_NEGFIRST = True, False



class Sentiment140LexiconUnigrams(NRCHashtagSentimentLexiconUnigrams):
  filename = os.path.join(_lexicon_data_dir, 'sentiment140_lexicon-unigrams.txt')



class Sentiment140LexiconBigrams(NRCHashtagSentimentLexiconUnigrams):
  filename = os.path.join(_lexicon_data_dir, 'sentiment140_lexicon-bigrams.txt')



class Sentiment140WithContextUnigrams(NRCHashtagSentimentLexiconUnigrams):
  filename = os.path.join(_lexicon_data_dir, 'sentiment140_lexicon_context-unigrams.txt')
  have_NEG, have_NEGFIRST = True, True



class Sentiment140WithContextBigrams(NRCHashtagSentimentLexiconUnigrams):
  filename = os.path.join(_lexicon_data_dir, 'sentiment140_lexicon_context-bigrams.txt')
  have_NEG, have_NEGFIRST = True, False



class YelpReviewsLexiconUnigrams(NRCHashtagSentimentLexiconUnigrams):
  filename = os.path.join(_lexicon_data_dir, 'yelp_reviews-unigrams.txt')
  have_NEG, have_NEGFIRST = True, True



class YelpReviewsLexiconBigrams(NRCHashtagSentimentLexiconUnigrams):
  filename = os.path.join(_lexicon_data_dir, 'yelp_reviews-bigrams.txt')
  have_NEG, have_NEGFIRST = True, False



class AmazonLaptopsReviewsLexiconUnigrams(NRCHashtagSentimentLexiconUnigrams):
  filename = os.path.join(_lexicon_data_dir, 'amazon_laptops_reviews-unigrams.txt')
  have_NEG, have_NEGFIRST = True, True



class AmazonLaptopsReviewsLexiconBigrams(NRCHashtagSentimentLexiconUnigrams):
  filename = os.path.join(_lexicon_data_dir, 'amazon_laptops_reviews-bigrams.txt')
  have_NEG, have_NEGFIRST = True, False



class MPQAEffectLexicon(AssociationScoreLexicon):
  filename = os.path.join(_lexicon_data_dir, 'mpqa_effect_wordnet.txt')

  def load_lexicon(self, filename=None, stemmed=False):
    if filename is None: filename = self.filename

    self._scores = {}

    with codecs.open(filename, 'r', 'utf-8') as f:
      for line in f:
        id_, eff, words, _ = line.strip().split(u'\t', 3)
        if eff == 'Null': continue
        eff = eff[0] + 'eff'

        for w in words.split(u','):
          w = w.replace(u'_', u' ')
          if stemmed: w = porter_stem(w)
          if w not in self._scores: self._scores[w] = {}
          self._scores[w][eff] = 1.0
        
      
    

    return u'Read {} items from {} (<{}>).'.format(len(self._scores), self.__class__.__name__, os.path.relpath(filename, _parentdir))
  



class MPQASubjectivityLexicon(AssociationScoreLexicon):
  filename = os.path.join(_lexicon_data_dir, 'mpqa_subjectivity_lexicon.txt')

  re_line = re.compile(r'type=(weaksubj|strongsubj) len=(\d+) word1=([\w\-]+) pos1=([\w]+) stemmed1=(n|y|1) priorpolarity=(positive|negative|neutral|both|weakneg|weakpos)')

  def load_lexicon(self, filename=None, stemmed=False):
    if filename is None: filename = self.filename

    self._scores = {}

    with codecs.open(filename, 'r', 'utf-8') as f:
      for line in f:
        m = self.re_line.match(line)
        if m is None: raise Exception('Unable to parse line in MPQASubjectivityLexicon: {}'.format(line.strip()))

        prior = m.group(6)
        if prior == 'neutral': continue

        w = m.group(3)
        if stemmed: w = porter_stem(w)

        strength = m.group(1)
        score = 1.0 if strength == 'strongsubj' else 0.5

        if w not in self._scores: self._scores[w] = {}
        if prior == 'both':
          self._scores[w]['+ve'] = score
          self._scores[w]['-ve'] = score
        elif prior == 'positive': self._scores[w]['+ve'] = score
        elif prior == 'negative': self._scores[w]['-ve'] = score
        elif prior == 'weakpos': self._scores[w]['+ve'] = score * 0.5
        elif prior == 'weakneg': self._scores[w]['-ve'] = score * 0.5
      
    

    return u'Read {} items from {} (<{}>).'.format(len(self._scores), self.__class__.__name__, os.path.relpath(filename, _parentdir))
  



class HarvardInquirerLexicon(AssociationScoreLexicon):
  filename = os.path.join(_lexicon_data_dir, 'harvard_inquirer.txt')

  def load_lexicon(self, filename=None, stemmed=False):
    if filename is None: filename = self.filename

    self._scores = {}

    re_hashtag = re.compile(r'\#\d+')

    with codecs.open(filename, 'r', 'utf-8') as f:
      for line in f:
        if '|' in line: prefix, _ = line.split(u'|', 1)
        else: prefix = line.strip()
        cols = prefix.strip().split()

        categories = {}
        for i, tok in enumerate(cols):
          if i == 0: w = re_hashtag.sub(u'', cols[0]).lower()
          elif i == 1: continue
          else: categories[tok.lower().strip('*')] = 1.0
        
        if stemmed: w = porter_stem(w)
        if w not in self._scores: self._scores[w] = categories
        else: self._scores[w].update(categories)
      
    

    return u'Read {} items from {} (<{}>).'.format(len(self._scores), self.__class__.__name__, os.path.relpath(filename, _parentdir))
  



class BingLiuLexicon(AssociationScoreLexicon):
  filenames = (os.path.join(_lexicon_data_dir, 'bingliu-positive.txt'), os.path.join(_lexicon_data_dir, 'bingliu-negative.txt'))

  def load_lexicon(self, filenames=None, stemmed=False):
    if filenames is None: filenames = self.filenames

    self._scores = {}

    for (score, filename) in zip((1.0, -1.0), filenames):
      with codecs.open(filename, 'r', 'utf-8') as f:
        for line in f:
          w = line.strip()
          if stemmed: w = porter_stem(w)
          if w not in self._scores: self._scores[w] = {'+ve': score}
        
      
    
    return u'Read {} items from {} (<{}>, <{}>).'.format(len(self._scores), self.__class__.__name__, os.path.relpath(filenames[0], _parentdir), os.path.relpath(filenames[1], _parentdir))
  



class AFINN111Lexicon(AssociationScoreLexicon):
  filename = os.path.join(_lexicon_data_dir, 'afinn-111.txt')

  def _parse_line(self, line):
    w, s = line.strip().split(u'\t', 1)
    return (w, '+ve', s)
  



class LoughranMcDonaldLexicon(AssociationScoreLexicon):
  filename = os.path.join(_lexicon_data_dir, 'loughran_mcdonald_master_dictionary.txt')

  def load_lexicon(self, filename=None, stemmed=False):
    if filename is None: filename = self.filename

    self._scores = {}
    with open(filename) as f:
      reader = csv.DictReader(f)
      categories = ['Negative', 'Positive', 'Uncertainty', 'Litigious', 'Constraining', 'Superfluous', 'Interesting', 'Modal', 'Irr_Verb', 'Syllables']

      for row in reader:
        w = row['Word'].lower()
        self._scores[w] = {}
        for cat in categories:
          s = float(row[cat])
          if s > 0.0:
            self._scores[w][cat] = s if cat == 'Modal' else 1.0
      
    

    return u'Read {} items from {} (<{}>).'.format(len(self._scores), self.__class__.__name__, os.path.relpath(filename, _parentdir))
  



class SentiWordNetLexicon(AssociationScoreLexicon):
  filename = os.path.join(_lexicon_data_dir, 'sentiwordnet-3.0.txt')

  def load_lexicon(self, filename=None, stemmed=False):
    if filename is None: filename = self.filename

    self._scores = {}
    re_sense = re.compile(r'\#\d+$')

    with codecs.open(filename, 'r', 'utf-8') as f:
      for line in f:
        if line.startswith('#'): continue
        pos, id_, pos_score, neg_score, synset, _ = line.split(u'\t', 5)
        if not id_: continue
        for w in synset.split():
          w = re_sense.sub('', w)
          if stemmed: w = porter_stem(w)
          self._scores[w] = {'pos': float(pos_score), 'neg': float(neg_score)}
        
      
    

    return u'Read {} items from {} (<{}>).'.format(len(self._scores), self.__class__.__name__, os.path.relpath(filename, _parentdir))
  



class SentimentSpecificWordEmbeddings(Lexicon):
  filename = os.path.join(_lexicon_data_dir, 'sswe-u.txt')

  def load_lexicon(self, filename=None, stemmed=False):
    if filename is None: filename = self.filename
    self._embeddings = {}
    self.dimensions = 50

    with codecs.open(filename, 'r', 'utf-8') as f:
      for line in f:
        cols = line.split(u'\t')
        assert len(cols) == self.dimensions + 1
        w = porter_stem(cols[0]) if stemmed else cols[0]
        self._embeddings[w] = np.array(cols[1:])
      
    

    self._unk_embedding = self._embeddings['<unk>']

    return u'Read {} embeddings from {} (<{}>).'.format(len(self._embeddings), self.__class__.__name__, os.path.relpath(filename, _parentdir))
  

  def __getitem__(self, w): return self._embeddings.get(w, self._unk_embedding)

