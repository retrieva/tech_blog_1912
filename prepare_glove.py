import gensim.downloader as api

from conll_dataset import ConllLoader

gensim_model = api.load("glove-wiki-gigaword-300") # 使用する分散表現

data = ConllLoader('/mnt/NER/corpus/CoNLL-2003/eng.train') # CoNLL2003訓練データパス
data.output_vocab(model, '/mnt/tech_blog/glove-wiki-300-connl') # 出力データのパス + prefix
