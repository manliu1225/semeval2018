import numpy as np
import collections
import nltk
from sklearn.cross_validation import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SpatialDropout1D, Dense, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from ..tweetokenize import Tokenizer
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import argparse
import codecs
import sklearn.utils.class_weight
from sklearn.metrics import accuracy_score
np.random.seed(42)

class BiLSTMClassifier(BaseEstimator, TransformerMixin):
    ''' test classification using lstm'''
    def __init__(self, MAX_VOCAB_SIZE = 2000, EPOCHS = 200, MAX_LENGTH = 50, EMBED_SIZE = 4000):
        # self.MAX_VOCAB_SIZE = MAX_LENGTH
        self.EPOCHS = EPOCHS
        self.MAX_LENGTH = MAX_LENGTH
        # self.EMBED_SIZE = EMBED_SIZE
    # def process_data(self, X, y = None):
    #     max_len = 0
    #     counter = collections.Counter()
    #     for i in xrange(len(X)):
    #         gettokens = Tokenizer()
    #         toks = gettokens.tokenize(X[i])
    #         for tok in toks:
    #             counter[tok] += 1
    #         max_len = max(max_len, len(toks))

    #     word2idx = collections.defaultdict(int)
    #     for idx, word in enumerate(counter.most_common(self.MAX_VOCAB_SIZE)):
    #         word2idx[word[0]] = idx + 2
    #     word2idx['UNK'] = 1
    #     word2idx['PAD'] = 0
    #     idx2word = {v: k for k, v in word2idx.items()}
    #     vocab_size = len(word2idx)

    #     # np.save('models/umich_word2idx_bidirectional_lstm.npy', word2idx)
    #     # np.save('models/umich_idx2word_bidirectional_lstm.npy', word2idx)

    #     context = {}
    #     context['maxlen'] = self.MAX_LENGTH
    #     # np.save('models/umich_context_bidirectional_lstm.npy', context)

    #     sx, sy = [], []
    #     for sentence in X:
    #         words = gettokens.tokenize(sentence.lower())
    #         wids = [word2idx[word] if word in word2idx else 1 for word in words]
    #         sx.append(wids)
    #     X = pad_sequences(sx, maxlen=self.MAX_LENGTH)
    #     if y: 
    #         for label in y: sy.append(int(label))
    #         y = np_utils.to_categorical(sy, 20)
    #     return X, y, vocab_size
    def set_params(self, **params):
        for key, value in params.items():
            if key.startswith('dropout1'): self.dropout1 = value
            if key.startswith('dropout2'): self.dropout2 = value
        return self
    # def batch_generator(self, X, batch_size, y=None,):
    #     shuffle_index = np.arange(np.shape(X)[0])
    #     while 1:
    #         np.random.shuffle(shuffle_index)
    #         X =  X[shuffle_index]
    #         if not y is None: 
    #             y = np_utils.to_categorical(y , 20)
    #             y =  y[shuffle_index]
    #         for i in xrange(np.ceil(1.0*X.shape[0]/batch_size).astype(int)):
    #             index_batch = shuffle_index[batch_size*i:batch_size*(i+1)]
    #             X_batch = X[index_batch].todense()
    #             X_batch = np.array(X_batch)
    #             X_batch = X_batch.reshape((X_batch.shape[0], 1, X_batch.shape[1]))
    #             # print 'load batch {}'.format(i)
    #             if not y is None:
    #                 y_batch = np.array(y[index_batch])
    #                 # print y_batch.shape
    #                 # print X_batch.shape
    #                 yield(X_batch,y_batch)
    #             else: yield X_batch
    def fit(self, X, y):
        # X, y, self.vocab_size = self.process_data(X, y)
        class_weight = sklearn.utils.class_weight.compute_class_weight('balanced',np.unique(y), y)
        self.class_weight_dict = dict(enumerate(class_weight))
        y = np_utils.to_categorical(y, 20)
        # self.class_weight_dict = {k: v-0.23 for k, v in self.class_weight_dict.items()}
        # # self.class_weight_dict = {0: 0.2620689655172414, 1: 0.47499999999999998, 2: 0.58461538461538465, 3: 1.52, 4: 1.2666666666666666, 5: 0.69090909090909092, 6: 1.52, 7: 0.69090909090909092, 8: 0.94999999999999996, 9: 0.84444444444444444, 10: 0.94999999999999996, 11: 1.52, 12: 2.5333333333333332, 13: 3.7999999999999998, 14: 1.8999999999999999, 15: 1.52, 16: 3.7999999999999998, 17: 7.5999999999999996, 18: 7.5999999999999996, 19: 0.94999999999999996}
        # print self.class_weight_dict
        self.model = Sequential()
        # self.model.add(Embedding(input_dim = self.vocab_size, output_dim=self.EMBED_SIZE, input_length = X.shape[1]))
        # self.model.add(SpatialDropout1D(0.2))
        print X.shape[1]
        print X.shape[2]
        self.model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2), input_shape=(X.shape[1], X.shape[2])))
        print self.model.output_shape
        # self.model.add(Bidirectional(LSTM(unit
        # s=64, dropout=0.2, recurrent_dropout=0.2), input_shape=(1, X.shape[1])))
        # self.model.add(Dense(32, activation='linear'))
        self.model.add(Dense(20, activation='softmax'))
        print self.model.output_shape
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        # self.model.fit_generator(generator=self.batch_generator(X = X, y = y, batch_size=self.BATCH_SIZE), samples_per_epoch = X.shape[0], epochs=self.EPOCHS, verbose=1)
        # self.model.fit_generator(generator=self.batch_generator(X = X, y = y, batch_size=self.BATCH_SIZE), steps_per_epoch = np.ceil(1.0*X.shape[0]/self.BATCH_SIZE), class_weight=self.class_weight_dict, epochs=self.EPOCHS, verbose=1)
        self.model.fit(X, y, class_weight=self.class_weight_dict, epochs=self.EPOCHS, verbose=1)
        return self
    def predict_proba(self, X):
        print 'predict ...'
        return self.model.predict(X)
        # return self.model.predict_generator(generator=self.batch_generator(X=X, batch_size=self.BATCH_SIZE), steps=np.ceil((1.0*X.shape[0])/self.BATCH_SIZE), use_multiprocessing=False, verbose=1)

    def predict(self, X):
        y_pred = []
        y_pred_li = self.predict_proba(X)
        for e in y_pred_li:
            y_pred.append(list(e).index(max(list(e))))
        return np.array(y_pred)

# X_train_file = '/Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_train_1.text'
# y_train_file  = '/Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_train_1.labels'
# X_test_file = '/Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_test_1.text'
# y_test_file = '/Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_test_1.labels'
# parser = argparse.ArgumentParser(description='lstm model.',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--X_train', help='input file')
# parser.add_argument('--y_train', help='input file')
# parser.add_argument('--X_test', help='input file')
# parser.add_argument('--y_test', help='input file')
# args = parser.parse_args()

# X_train_file = args.X_train
# y_train_file  = args.y_train
# X_test_file = args.X_test
# y_test_file = args.y_test

# with open(X_train_file, 'r') as f1:
#     X_train = map(lambda x: x.strip(), f1.readlines())
# with open(y_train_file, 'r') as f2:
#     y_train = map(lambda x: x.strip(), f2.readlines())
# assert len(X_train) == len(y_train)
# with open(X_test_file, 'r') as f3:
#     X_test = map(lambda x: x.strip(), f3.readlines())
# with open(y_test_file, 'r') as f4:
#     y_test = map(lambda x: x.strip(), f4.readlines())
#     y_test = map(lambda x:int(x) , y_test)

# lstm_cfl = LSTMClassifier()
# lstm_cfl.fit(X_train, y_train)
# print('predict:')
# y_pred = lstm_cfl.predict(X_test)
# print(classification_report(y_test, y_pred))
# print(lstm_cfl.predict_proba(X_test)[0])
