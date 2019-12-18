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

BATCH_SIZE = 5 # 訓練時のバッチサイズ
EPOCHS = 5 # 訓練時のエポック数
EMBEDDING_DIM = 300 # 学習済み単語分散表現の次元数
MAX_LENGTH = 32 # 学習データの最大系列長より大きい必要がある

# CoNLL 2003 データセットから(stringのlist, stringのlist)の形式で、各種データを読み込みます。
X_train, y_train = ConllLoader('/mnt/NER/corpus/CoNLL-2003/eng.train').get_list()
X_dev, y_dev = ConllLoader('/mnt/NER/corpus/CoNLL-2003/eng.testa').get_list()
X_test_original, y_test = ConllLoader('/mnt/NER/corpus/CoNLL-2003/eng.testb').get_list()

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
y_test = encode_and_pad_data(y_test, label_token_encoder)

# tf.keras.Model　を継承してモデルを定義する。
class BilstmCRF(tf.keras.Model):
    def __init__(self, num_tags, token_encoder, label_encoder, embedding_list):
        super().__init__()
        self.num_tags = num_tags
        self.token_encoder = token_encoder
        self.label_encoder = label_encoder

        self.embedding_layer = Embedding(input_dim=token_encoder.vocab_size, output_dim=EMBEDDING_DIM, 
                                         embeddings_initializer=tf.keras.initializers.Constant(embedding_list), mask_zero=True)
        self.bilstm_layer = Bidirectional(LSTM(200, return_sequences=True), merge_mode='ave')
        self.dense_layer = Dense(num_tags, activation='softmax')
        self.crf_layer = CRF(num_tags, name='crf_layer')

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


# 以下でモデルの定義
# 独自クラスによってモデルを作る。
model = BilstmCRF(n_labels, word_token_encoder, label_token_encoder, glove)
# モデルの初期化を行います。ここで、誤差関数にBilstmCRFのcrf_lossメソッドを指定します。
model.compile(optimizer='adam', loss=model.crf_loss, metrics=['accuracy'])
# TensorBoard用のログの保存場所を定義します。
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# ここで、学習データを渡すことで、学習を行います。
model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=EPOCHS,  
    batch_size=BATCH_SIZE, 
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)])

# モデルの保存
model.save('saved_model/model_subclass.h5')

# 予測ラベルの取得
y_pred = model.predict([X_test][:5])

# 予測結果の表示
for i in range(5):
    print('元の文:', X_test_original[i])    
    print('正解ラベル:', label_token_encoder.decode(y_test[i]))    
    print('予測ラベル:', label_token_encoder.decode(y_pred[i]))  
    print()  
    with open('ner_results.txt', 'a') as f:
        f.write('元の文:'+X_test_original[i]+'\n')
        f.write('正解ラベル:'+label_token_encoder.decode(y_test[i])+'\n')
        f.write('予測ラベル:'+label_token_encoder.decode(y_pred[i])+'\n\n')


