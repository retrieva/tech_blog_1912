import datetime

import numpy as np
import tensorflow as tf 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow_addons.layers import CRF 
from tensorflow_addons.losses import ConditionalRandomFieldLoss
import tensorflow_datasets as tfds

from conll_dataset import ConllLoader

EMBEDDING_DIM = 300
MAX_LENGTH = 32


glove = np.concatenate([np.zeros(EMBEDDING_DIM)[np.newaxis], np.load('glove-wiki-300-connl.npy'), np.zeros(EMBEDDING_DIM)[np.newaxis]], axis=0)


X_train, y_train = ConllLoader('/mnt/NER/corpus/CoNLL-2003/eng.train').get_list()
X_dev, y_dev = ConllLoader('/mnt/NER/corpus/CoNLL-2003/eng.testa').get_list()
X_test, y_test = ConllLoader('/mnt/NER/corpus/CoNLL-2003/eng.testb').get_list()

with open('glove-wiki-300-connl.vocab') as f:
    vocab_list = f.readlines()

def preprocess_data(data, encoder):
    data = [encoder.encode(sequence) for sequence in data]
    data = pad_sequences(data, maxlen=MAX_LENGTH, dtype='int32', padding='post', value=0)
    return data

token_encoder = tfds.features.text.TokenTextEncoder(vocab_list)

X_train = preprocess_data(X_train, token_encoder)
X_dev = preprocess_data(X_dev, token_encoder)
X_test = preprocess_data(X_test, token_encoder)

tag_list = list(set([tag for tag_in_sentence in y_train for tag in tag_in_sentence.split()]))
n_tags = len(tag_list)
label_encoder = tfds.features.text.TokenTextEncoder(tag_list)

y_train = preprocess_data(y_train, label_encoder)
y_dev = preprocess_data(y_dev, label_encoder)
y_test = preprocess_data(y_test, label_encoder)

class BilstmCRF(tf.keras.Model):
    def __init__(self, num_tags, token_encoder, label_encoder, embedding_list):
        super().__init__()
        self.num_tags = num_tags
        self.token_encoder = token_encoder
        self.label_encoder = label_encoder

        self.embedding_layer = Embedding(input_dim=token_encoder.vocab_size, output_dim=EMBEDDING_DIM, 
                                         embeddings_initializer=tf.keras.initializers.Constant(embedding_list), mask_zero=True)
        self.bilstm_layer = Bidirectional(LSTM(400, return_sequences=True), merge_mode='ave')
        self.dense_layer = Dense(n_tags, activation='softmax')
        self.crf_layer = CRF(n_tags, name='crf_layer')

    @tf.function(autograph=True) # autograph=Falseでグラフモードへの変換を停止
    def call(self, inputs):
        x = self.embedding_layer(inputs)
        mask = x._keras_mask
        x = self.bilstm_layer(x, mask=mask)
        x = self.dense_layer(x)
        x = self.crf_layer(x, mask=tf.cast(mask, tf.int32)) 

        return x

    def crf_loss(self, true_label, pred_label):
        loss = self.crf_layer.get_loss(true_label, pred_label)
        return loss



model = BilstmCRF(n_tags, token_encoder, label_encoder, glove)
model.compile(optimizer='adam', loss=model.crf_loss, metrics=['accuracy'])
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=5,  
    batch_size=5, 
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'), 
               tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)])
model.evaluate(X_test, y_test)

