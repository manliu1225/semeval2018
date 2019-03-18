from tweets import load_instances, BaseSentimentClassifier, PureTransformer, ListCountVectorizer, BasicTokenizer, NGramTransformer, CharNGramTransformer, CMUArkTweetPOSTagger, CMUArkTweetBrownClusters, LowercaseTransformer, Negater
from argparse import ArgumentParser
import codecs
from tweets.nrc import NRCSentimentClassifier
import pickle
from tweets.lstm import BiLSTMSentimentClassifier
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer, f1_score
from tweets.gumltlt import GUMLTLTSentimentClassifier
from tweets.cool import CoolSentimentClassifier
# from sklearn.ensemble import VotingClassifier
# from tweets.ensemble import VotingEnsembleClassifier
from mlxtend.classifier import EnsembleVoteClassifier
# from tweets.gbdt_allfeatures import GBDTSentimentClassifier
from tweets.lightgbm_allfeatures import GBDTSentimentClassifier
from sklearn.externals import joblib
# import os

parser = ArgumentParser(description='test')
parser.add_argument('--X_train', type=open,  metavar='file', default=None, required=True, help='List of files to use as training data.')
parser.add_argument('--y_train', type=open, metavar='file', default=None, help='List of files to use as test data. Can be empty if only cross validation experiments are desired.')
parser.add_argument('--X_test', type=open,  metavar='file', default=None, required=True, help='List of files to use as training data.')
parser.add_argument('--y_test', type=open, metavar='file', default=None, help='List of files to use as test data. Can be empty if only cross validation experiments are desired.')
parser.add_argument('--pred_f', help='predicted y.')
parser.add_argument('--estimator', default='gbdt', help='give the estimator you want to use.')
parser.add_argument('--save_model', type=str, help='Save model')
parser.add_argument('--load_model', type=str, help='Load model')
parser.add_argument('--num_leaves', default=64, type=int, help='parameters for classifier')
parser.add_argument('--n_estimators', default=300, help='parameters for classifier')

args = parser.parse_args()

X_train, y_train = load_instances(args.X_train, args.y_train)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test, y_test = load_instances(args.X_test, args.y_test)
X_test = np.array(X_test)
y_test = np.array(y_test)

nrc = NRCSentimentClassifier()

gumltlt = GUMLTLTSentimentClassifier()

cool = CoolSentimentClassifier()

gbdt = GBDTSentimentClassifier(classifier__num_leaves=args.num_leaves, classifier__n_estimators=args.n_estimators)
# gbdt = GBDTSentimentClassifier(classifier__num_leaves=20, classifier__n_estimators=700, classifier__min_data=1)


lstmclf = BiLSTMSentimentClassifier()

# eclf1 = EnsembleVoteClassifier(clfs=[nrc, gumltlt, cool],  voting='soft')
# eclf1 = VotingEnsembleClassifier(
#   estimators = dict(
#       nrc = NRCSentimentClassifier(),
#       gumltlt = GUMLTLTSentimentClassifier(),
#       cool = CoolSentimentClassifier()))
# print 'estimator is {}'.format(args.estimator)



if args.load_model: estimator = joblib.load(args.load_model) 
else: 
	if args.estimator == 'lstmclf': estimator = lstmclf
	else: estimator = gbdt
	# print(estimator.get_params().keys())
	# estimator = eval(args.estimator)
	estimator.fit(X_train, y_train)

# parameters = {'classifier__num_leaves':[16,32,64, 128], 'classifier__n_estimators':[100, 300, 700, 1000], 'classifier__min_data':[5]}
# macrof1_scorer = make_scorer(f1_score, average='macro')
# estimator = GridSearchCV(gbdt, parameters, scoring=macrof1_scorer,n_jobs=16, verbose=3, cv=3)
# estimator.fit(X_train, y_train)
# bp = dictestimator.best_params_
# print('best model is {}'.format(estimator.best_params_))


if args.save_model:
	joblib.dump(estimator, args.save_model)
	# joblib.dump(estimator, 'gbdt_leaves{}_nestimators{}.model'.format(args.num_leaves, args.n_estimators))
y_pred = estimator.predict(X_test)
# with open('data/{}/{}_pred_labels.txt'.format(str(estimator).strip("( )"),str(args.X_test).split('.')[0].split('/')[-1]), 'w') as f:
with open(args.pred_f, 'w') as f:
    for e in y_pred:
        # print e
        f.write('{}'.format(e))
        f.write('\n')

# dict_para = estimator.named_steps['classifier'].get_params()
# print('parameters are {}'.format(', '.join(['{}:{}'.format(k, v) for k,v in dict_para.items()])))
print(classification_report(y_test, y_pred))
