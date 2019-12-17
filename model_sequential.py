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

glove = np.concatenate([np.zeros(EMBEDDING_DIM)[np.newaxis], np.load('glove-wiki-300-connl.npy'), 
        np.zeros(EMBEDDING_DIM)[np.newaxis]], axis=0)


X_train, y_train = ConllLoader('/mnt/NER/corpus/CoNLL-2003/eng.train').get_list()
X_dev, y_dev = ConllLoader('/mnt/NER/corpus/CoNLL-2003/eng.testa').get_list()
X_test_original, y_test = ConllLoader('/mnt/NER/corpus/CoNLL-2003/eng.testb').get_list()

with open('glove-wiki-300-connl.vocab') as f:
    vocab_list = f.readlines()


def preprocess_data(data, encoder):
    data = [encoder.encode(sequence) for sequence in data]
    data = pad_sequences(data, maxlen=MAX_LENGTH, dtype='int32', padding='post', value=0)
    return data

token_encoder = tfds.features.text.TokenTextEncoder(vocab_list)

X_train = preprocess_data(X_train, token_encoder)
X_dev = preprocess_data(X_dev, token_encoder)
X_test = preprocess_data(X_test_original, token_encoder)

tag_list = list(set([tag for tag_in_sentence in y_train for tag in tag_in_sentence.split()]))
n_tags = len(tag_list)
label_encoder = tfds.features.text.TokenTextEncoder(tag_list)

y_train = preprocess_data(y_train, label_encoder)
y_dev = preprocess_data(y_dev, label_encoder)
y_test = preprocess_data(y_test, label_encoder)



model = tf.keras.Sequential()
model.add(Embedding(input_dim=token_encoder.vocab_size, output_dim=EMBEDDING_DIM, 
                    embeddings_initializer=tf.keras.initializers.Constant(glove), 
                    input_shape=(MAX_LENGTH, ), ))#mask_zero=True))
model.add(Bidirectional(LSTM(200, return_sequences=True), merge_mode='ave'))
model.add(Dense(n_tags, activation='softmax'))
model.add(CRF(n_tags, name='crf_layer'))


model.compile(optimizer='adam', loss=ConditionalRandomFieldLoss())
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=5,  
    batch_size=5, 
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'), 
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)])
model.evaluate(X_test, y_test)
model.save('saved_model/model_sequential.h5')


y_pred = model.predict([X_test][:5])

for i in range(5):
    print('元の文:', X_test_original[i])    
    print('正解ラベル:', label_encoder.decode(y_test[i]))    
    print('予測ラベル:', label_encoder.decode(y_pred[i]))  
    print()  
    with open('ner_results.txt', 'a') as f:
        f.write('元の文:'+X_test_original[i]+'\n')
        f.write('正解ラベル:'+label_encoder.decode(y_test[i])+'\n')
        f.write('予測ラベル:'+label_encoder.decode(y_pred[i])+'\n\n')