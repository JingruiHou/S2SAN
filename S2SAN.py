#! /usr/bin/env python
from __future__ import print_function
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers

from nltk import tokenize
import pickle
import time
import pandas as pd
from keras import initializers
import os
from keras.engine.topology import Layer
from keras import backend as K
from sklearn.model_selection import StratifiedKFold


class Word_Attention(Layer):

    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(Word_Attention, self).__init__()

    def build(self, input_shape):
        print(len(input_shape))
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(Word_Attention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class Attention(Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x

        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        #计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)
        #输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


SEED = 9

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 400
MAX_SENTS = 15

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

reviews = []
tokenized_text = []

df = pd.read_csv('../data/amazon_cd.csv')
X = df['reviewText']
X = [str(x) for x in X]

Y = np.asarray(df['overall'], dtype=np.int)
Y = [0 if star <= 2 else 1 if star == 3 else 2 for star in list(Y)]
# Y = to_categorical(np.asarray(Y))

for txt in list(X):
    sentences = tokenize.sent_tokenize(txt)
    reviews.append(sentences)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(list(X))

for sentences in reviews:
    tokenized_sentences = tokenizer.texts_to_sequences(sentences)
    tokenized_sentences = tokenized_sentences[:MAX_SENTS]
    tokenized_text.append(tokenized_sentences)

X = np.zeros((len(tokenized_text), MAX_SENTS, MAX_SEQUENCE_LENGTH), dtype='int32')
for i in range(len(tokenized_text)):
    sentences = tokenized_text[i]
    seq_sequences = pad_sequences(sentences, maxlen=MAX_SEQUENCE_LENGTH)
    for j in range(len(seq_sequences)):
        X[i, j] = seq_sequences[j]


print(MAX_NB_WORDS)

print('Build model...')


def build_hfan(maxlen=MAX_SEQUENCE_LENGTH, max_sent_len=MAX_SENTS, max_words=MAX_NB_WORDS, embedding_dim=EMBEDDING_DIM, classification_type=3):
    S_inputs = layers.Input(shape=(maxlen,), dtype='int32')
    O_seq = layers.Embedding(max_words, embedding_dim, input_length=maxlen)(S_inputs)
    O_seq = layers.Bidirectional(layers.GRU(100, return_sequences=True))(O_seq)
    O_seq = Word_Attention(100)(O_seq)
    sentences_model = models.Model(inputs=S_inputs, outputs=O_seq)
    sentences_model.summary()

    review_input = layers.Input(shape=(max_sent_len, maxlen), dtype='int32')
    review_encoder = layers.TimeDistributed(sentences_model)(review_input)
    l_lstm_sent = Attention(1, 32)([review_encoder, review_encoder, review_encoder])
    l_att_sent = layers.GlobalAveragePooling1D()(l_lstm_sent)
    preds = layers.Dense(classification_type, activation='softmax')(l_att_sent)
    model = models.Model(review_input, preds)
    return model


file_name = 'S2SAN'
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

index = 0
for train, test in kfold.split(X, Y):
    x_train = X[train]
    y_train = to_categorical(np.asarray(Y))[train]
    x_test = X[test]
    y_test = to_categorical(np.asarray(Y))[test]
    model = build_hfan()
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    start = time.clock()
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        nb_epoch=5, batch_size=50)
    total_training_time = time.clock() - start
    index += 1
    with open("../saves/model_fitting_log", 'a') as f:
        f.write('Total training time of {}_{} is {}\n'.format(file_name, index, total_training_time))

    model.save(filepath="../saves/{}_{}_{}.h5".format(file_name, index, time.time()))
    history_dict = history.history
    with open("../saves/{}_{}_{}.dic".format(file_name, index, time.time()), "wb") as f:
        pickle.dump(history_dict, f)
    del model
