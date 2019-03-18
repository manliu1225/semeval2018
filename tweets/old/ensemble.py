from collections import defaultdict
import logging

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


class VotingEnsembleClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, estimators={}):
    self.estimators_ = estimators

    self.estimator_classes_map_ = {}
    for i, (key, estimator) in enumerate(self.estimators_.iteritems()):
      self.estimator_classes_map_[key] = dict([(c, j) for j, c in enumerate(getattr(estimator, 'classes_', []))])
    print self.estimator_classes_map_
    self.classes_ = np.array(range(20))
    logger.debug('VotingEnsembleClassifier initialized.')
  #end def

  def fit(self, X, y):
    for i, (key, estimator) in enumerate(self.estimators_.iteritems()):
      estimator_type_name = type(estimator).__name__
      logger.debug('Training estimator `{}` ({})'.format(key, estimator_type_name))
      estimator.fit(X, y)  # train using this .75 subset
      logger.debug('Done training estimator `{}` ({})'.format(key, estimator_type_name))

      self.estimator_classes_map_[key] = dict([(c, j) for j, c in enumerate(estimator.classes_)])
      print 'estimator_classes_map_ in fit {}'.format(self.estimator_classes_map_)
    #end for

    return self
  #end def

  def predict(self, X, debug=False):
    return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
  #end def

  def predict_log_proba(self, X):
    return np.log(self.predict_proba(X))
  #end def

  def predict_proba(self, X):
    y_proba_all = np.zeros((X.shape[0] if hasattr(X, 'shape') else len(X), 20))
    for i, (key, estimator) in enumerate(self.estimators_.iteritems()):
      y_proba = estimator.predict_proba(X)
      print key
      print 'estimator_classes_map_ in predict_proba {}'.format(self.estimator_classes_map_)
      y_proba_all += y_proba[:, [self.estimator_classes_map_[key][class_name] for class_name in estimator.classes_]]
      # y_proba_all[:, 0] += y_proba[:, self.estimator_classes_map_[key][-1]]
      # y_proba_all[:, 1] += y_proba[:, self.estimator_classes_map_[key][0]]
      # y_proba_all[:, 2] += y_proba[:, self.estimator_classes_map_[key][1]]
      print 'y_proba_all {}'.format(y_proba_all)
    #end for

    return y_proba_all / len(self.estimators_)
  #end def

  def predict_all_proba(self, X):
    # L = X.shape[0] if hasattr(X, 'shape') else len(X)
    estimator_proba = {}
    for i, (key, estimator) in enumerate(self.estimators_.iteritems()):
      y_proba = estimator.predict_proba(X)
      print 'key in predict_all_proba {}'.format(key)
      print 'estimator_classes_map_ {}'.format(self.estimator_classes_map_)
      estimator_proba[key] = y_proba[:, [self.estimator_classes_map_[key][class_name] for class_name in estimator.classes_]]
    #end for
      print 'shape of estimator_proba {}'.format(estimator_proba)

    estimator_proba['ensemble'] = sum(estimator_proba.values()) / len(self.estimators_)
    return estimator_proba
  #end def

  def transform(self, X):
    return self.predict(X)
  #end def

  def get_params(self, deep=True):
    if deep:
      params = {}
      for estimator_key, estimator in self.estimators_.iteritems():
        for key, value in estimator.get_params(deep).iteritems():
          params[estimator_key + '__' + key] = value

      return params
    #end if

    return {}
  #end def

  def set_params(self, **params):
    estimator_params = defaultdict(dict)
    for key, value in params.iteritems():
      estimator_key, param_key = key.split('__', 1)
      if estimator_key != 'ensemble' and estimator_key not in self.estimators_: raise Exception('The key `{}` does not exist in ensemble.'.format(estimator_key))
      estimator_params[estimator_key][param_key] = value
    #end for

    for estimator_key, p in estimator_params.iteritems():
      if estimator_key == 'ensemble': continue
      else: self.estimators_[estimator_key].set_params(**p)
    #end for
    return self
  #end def
#end class
