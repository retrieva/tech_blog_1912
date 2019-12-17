tf, tfds, np = None, None, None 

class ConllLoader:
    def __init__(self, data_path, drop_docstart=True, max_length=None, padding='<PAD>'):
        self.max_length = max_length
        self.padding = padding
        self.sentence_lengths = []
        self.mask = []
        self.vocab = []
        self.embeddings = []

        self.sentences, self.labels = [], []
        sentence, labels_for_the_sentence = [], []
        
        self.char_sentences = []
        char_sentence = []

        with open(data_path, 'r') as f:
            connl_format_data = f.read().splitlines() 
        

        for line in connl_format_data:
            if line.strip() == '':
                sentence_length = len(sentence)
                self.sentence_lengths.append(sentence_length)
                self.sentences.append(sentence)
                self.labels.append(labels_for_the_sentence)
                sentence, labels_for_the_sentence = [], []

                continue
            word, _, _ , label = line.split()
            sentence.append(word)
            labels_for_the_sentence.append(label)
            
        if self.max_length:
            for i, word_token_list in enumerate(self.sentences):
                if len(self.sentences[i]) > self.max_length:
                    self.sentences[i] = self.sentences[i][:self.max_length]
                    self.mask.append([True] * self.max_length)
                else:
                    self.sentences[i] += [self.padding] * (self.max_length - len(self.sentences[i]))
                    self.mask.append([True] * len(self.sentences[i]) + [False] * (self.max_length - len(self.sentences[i])))


        for i, word_token_list in enumerate(self.sentences): 
            self.sentences[i] = ' '.join(self.sentences[i])
            self.labels[i] = ' '.join(self.labels[i])

        if drop_docstart:
            self.sentences = self.sentences[1:]
            self.labels = self.labels[1:]
            
    def get_list(self):

        return self.sentences, self.labels

    def get_ndarray(self):
        global np 
        if np is None:
            import numpy as np 
        
        sentence_array = np.array(self.sentences)
        label_array = np.array(self.labels)

        return sentence_array, label_array 


    def get_dataset(self):
        global tf, tfds 
        if tf is None:
            import tensorflow as tf 
            import tensorflow_datasets as tfds 

        sentence_dataset = tf.data.Dataset.from_tensor_slices(self.sentences)
        label_dataset = tf.data.Dataset.from_tensor_slices(self.labels)

        return sentence_dataset, label_dataset

    def output_vocab(self, gensim_model, output_file_path):
        global np 
        if np is None:
            import numpy as np 
        self.vocab = []
        self.embeddings = []

        with open(output_file_path + '.vocab', 'w') as f:
            for sentence in self.sentences:
                words = sentence.split() 
                for word in words:
                    if word not in self.vocab:
                        self.vocab.append(word)
                        f.write(word + '\n')
                        if word in gensim_model.wv:
                            self.embeddings.append(gensim_model.wv[word])
                        else:
                            self.embeddings.append(np.zeros(gensim_model.wv.vector_size))
        
        np.save(output_file_path + '.npy', self.embeddings)

        
            
        


if __name__ == '__main__':
    test = ConllLoader('/mnt/NER/corpus/CoNLL-2003/eng.train') 
    print(test.get_ndarray()[0][0])
    print(test.get_ndarray()[1][0])