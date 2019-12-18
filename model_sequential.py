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


# 以下でモデルの定義
# Sequential　APIを使ってモデルの定義
model = tf.keras.Sequential()
# Embedding レイヤーで、 単語インデックスをそれに対応する単語分散表現に置き換えます。
model.add(Embedding(input_dim=word_token_encoder.vocab_size, output_dim=EMBEDDING_DIM, 
                    embeddings_initializer=tf.keras.initializers.Constant(glove), 
                    input_shape=(MAX_LENGTH, )))#　mask_zero=True　は上手くいかない
# 出力次元数200のBiLSTMレイヤーを追加します。
# LSTMクラスのイニシャライザにreturn_sequences=Trueを渡すことで、
# 全ての単語に対する予測値を取り出せます。
# return_sequences=Falseだと1文に対して1つの予測値のみの出力となります。
model.add(Bidirectional(LSTM(200, return_sequences=True), merge_mode='ave'))
# 出力次元数がタグの種類数の全結合層を追加します。
# また、活性化関数をsoftmaxにします。
# これは、CRFレイヤーが各ラベルに対する予測の確信度を要求するためです。
model.add(Dense(n_labels, activation='softmax'))
# 最後にCRFレイヤーを追加します。
model.add(CRF(n_labels, name='crf_layer'))

# モデルの初期化を行います。ここで、誤差関数にConditionalRandomFieldLossを指定します。
model.compile(optimizer='adam', loss=ConditionalRandomFieldLoss())
# TensorBoard用のログの保存場所を定義します。
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# ここで、学習データを渡すことで、学習を行います。
model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=EPOCHS,  
    batch_size=BATCH_SIZE, 
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)])
# モデルの保存
model.save('saved_model/model_sequential.h5')

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

