from config import config
import zipfile
import numpy as np
import torch
from torch.autograd import Variable

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.found_words = 0
        self.new_words = 0

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class embeddings:
    def __init__(self):
        pass

    def load_vocabulary_word_vectors(self,path,name,dimension):
        if config.USE_GLOVE:
            print("load the GloVe dataset sample that matches to our vocabulary\n")

            zf = zipfile.ZipFile(path)
            glove_ds_sample = zf.read(name).decode('utf8').splitlines()

            wordvecs = {}
            for line in glove_ds_sample:
                l = line.split()
                vec_idx = len(l) - dimension
                word = " ".join(l[:vec_idx])
                wordvecs[word] = np.asarray(l[vec_idx:], dtype='float')

            zf.close()

            print('loaded:' + str(len(wordvecs.keys())))
            print("initialize the vocabulary words that are not in Glove to the 0 vector\n")
            print("done!\n")
            self.wordvecs = wordvecs
        else:
            self.wordvecs = []

        return

    def sentence_to_embeddings(self, lang, sentence):
        indexes = []
        for token, j in zip(sentence, range(len(sentence))):
            if type(token) == list:
                token_ind = []
                for part, i in zip(token, range(len(token))):
                    if part.lower() in self.wordvecs and (i == 0 or i == 2) and j > 2:
                        token_ind.append(self.wordvecs[part.lower()])
                        lang.found_words += 1
                    elif (i == 0 or i == 2) and j > 2:
                        token_ind.append(lang.word2index[part])
                        lang.new_words += 1
                    else:
                        token_ind.append(lang.word2index[part])
                indexes.append(token_ind)

        indexes.append([config.EOS_token, config.EOS_token, config.EOS_token])

        for i in range(len(indexes)):
            for j in range(len(indexes[i])):
                if type(indexes[i][j]) == int:
                    indexes[i][j] = Variable(torch.LongTensor([indexes[i][j]]))
                else:
                    indexes[i][j] = Variable(torch.FloatTensor(indexes[i][j]), requires_grad=False)


        if config.use_cuda:
            for i in range(len(indexes)):
                for j in range(len(indexes[i])):
                    indexes[i][j] = indexes[i][j].cuda()

        return indexes

