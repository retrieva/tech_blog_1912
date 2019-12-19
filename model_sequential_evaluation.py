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

EMBEDDING_DIM = 300 # 学習済み単語分散表現の次元数
MAX_LENGTH = 32 # 学習データの最大系列長より大きい必要がある

# CoNLL 2003 データセットから(stringのlist, stringのlist)の形式で、各種データを読み込みます。
X_train, y_train = ConllLoader('/mnt/NER/corpus/CoNLL-2003/eng.train').get_list()
X_dev, y_dev = ConllLoader('/mnt/NER/corpus/CoNLL-2003/eng.testa').get_list()
X_test_original, y_test_original = ConllLoader('/mnt/NER/corpus/CoNLL-2003/eng.testb').get_list()

# 以下で学習済み分散表現の読み込みを行います。
# これによって、訓練データ以外の大規模コーパスから得た単語の意味情報を考慮できます。
# 0番目はパディング、 (語彙数+1)番目は未知語に割り当てられます。
glove = np.concatenate([np.zeros(EMBEDDING_DIM)[np.newaxis], np.load('../glove-wiki-300-connl.npy'), 
        np.zeros(EMBEDDING_DIM)[np.newaxis]], axis=0)

# 学習済み単語分散表現の語彙リストを読み込みます。
with open('../glove-wiki-300-connl.vocab') as f:
    vocab_list = f.readlines()

# スペース区切り単語列を単語インデックス列に変換するエンコーダを定義。
word_token_encoder = tfds.features.text.TokenTextEncoder(vocab_list)

# 全ラベルリスト　と ラベル数　の取得
label_list = list(set([label for label_in_sentence in y_train for label in label_in_sentence.split()]))
n_labels = len(label_list)

# スペース区切りラベル列を単語インデックス列に変換するエンコーダを定義。
label_token_encoder = tfds.features.text.TokenTextEncoder(label_list)

def encode_and_pad_data(sequence_list, encoder):
    # strのlistを単語インデックスのlistのlistに変換した後、
    # MAX_LENGTHに満たない文に対して、0でパディングを行う。
    enceded_sequences = [encoder.encode(sequence) for sequence in sequence_list]
    encoded_and_padded_sequences = pad_sequences(enceded_sequences, maxlen=MAX_LENGTH, dtype='int32', padding='post', value=0)
    return encoded_and_padded_sequences


# 以下でエンコードとパディングを行います。
X_train = encode_and_pad_data(X_train, word_token_encoder)
X_dev = encode_and_pad_data(X_dev, word_token_encoder)
X_test = encode_and_pad_data(X_test_original, word_token_encoder)
y_train = encode_and_pad_data(y_train, label_token_encoder)
y_dev = encode_and_pad_data(y_dev, label_token_encoder)
y_test = encode_and_pad_data(y_test_original, label_token_encoder)


# モデルの読み込み
model = tf.keras.models.load_model('../saved_model/model_sequential.h5')

# 予測ラベルの取得
y_pred = model.predict(X_test)


f = open('result_connl_format.txt', 'w')
# 予測結果の表示
for i in range(len(X_test_original)):
    sentence = X_test_original[i].split()
    sentence_length = len(sentence)
    true_labels = y_test_original[i].split()
    pred_labels = label_token_encoder.decode(y_pred[i]).split()

    if len(pred_labels) < sentence_length:
        pred_labels += ['O'] * (sentence_length - len(pred_labels))
    elif len(pred_labels) > sentence_length:
        pred_labels = pred_labels[:sentence_length]
    # if len(true_labels) < sentence_length:
    #     true_labels += ['O'] * (sentence_length - len(true_labels))
    # elif len(true_labels) > sentence_length:
    #     true_labels = true_labels[:sentence_length]
    
    for j in range(sentence_length):
       
        f.write(sentence[j] + ' unknown ' + true_labels[j] + ' ' + pred_labels[j] + '\n')
    f.write('\n')
f.close()