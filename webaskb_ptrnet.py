from config import config
from io import open
import numpy as np
import json

import torch
from torch.autograd import Variable

from Net.run import NNRun
from common.embeddings import Lang
from config import Config
from common.embeddings import embeddings
from Models.webaskb_ptrnet import WebAsKB_PtrNet_Model

class WebAsKB_PtrNet():
    def __init__(self):
        # Embeddings
        self.embed = embeddings()
        self.embed.load_vocabulary_word_vectors(config.glove_50d,'glove.6B.50d.txt',50)

    # Load Data
    def prepareData(self, filename, is_training_set, input_lang=None):
        if input_lang is None:
            input_lang = Lang('input')

        with open(filename, 'r') as outfile:
            split_dataset = json.load(outfile)

        print("Read %s sentence pairs" % len(split_dataset))
        print("Counting words...")

        input_lang.addWord('None')
        pairs = []
        for question in split_dataset:
            # training is done using only composition and conjunction examples
            if is_training_set and question['comp'] != 'composition' and question['comp'] != 'conjunction':
                continue

            x = [['None', 'None'], ['composition', 'None'], ['conjunction', 'None']]
            y = []
            aux_data = question

            if len(question['sorted_annotations'])>config.MAX_LENGTH-4:
                continue

            # dynamically building the input language
            input_lang.addWord('composition')
            input_lang.addWord('conjunction')
            input_lang.addWord('None')

            for token in question['sorted_annotations']:
                x.append([token['dependentGloss'],token['dep']])
                input_lang.addWord(token['dependentGloss'])
                input_lang.addWord(token['dep'])
            # returns embeded data (also converts to Variables tokens that were not found in Glove)
            x = self.embed.sentence_to_embeddings(input_lang, x)

            # adding actions to ouptuts
            if question['comp'] == 'composition':
                y.append(1)
            elif question['comp'] == 'conjunction':
                y.append(2)

            # adding split points to ouputs
            if question['p1'] == question['p1'] and question['p1'] is not None:
                y.append(int(question['p1']) + 3)
                if question['p2'] == question['p2']:
                    y.append(int(question['p2']) + 3)
                else:
                    y.append(0)

            if config.use_cuda:
                y = Variable(torch.LongTensor(y).view(-1, 1)).cuda()
            else:
                y = Variable(torch.LongTensor(y).view(-1, 1))

            pairs.append({'x':x,'y':y,'aux_data':aux_data})

        # shuffling the X,Y pairs
        print ("total number of pair:" + str(len(pairs)))
        np.random.seed(5)
        pairs = [pairs[i] for i in np.random.permutation(len(pairs))]

        return input_lang, pairs

    def load_data(self):
        # we always read the training data - to create the language index in the same order.
        self.input_lang, self.pairs_train = self.prepareData(config.noisy_supervision_dir + 'train.json',is_training_set=True)
        self.input_lang, self.pairs_dev = self.prepareData(config.noisy_supervision_dir + config.EVALUATION_SET + '.json', \
                                                           is_training_set=False , input_lang=self.input_lang)

    def init(self):
        # define batch training scheme
        model = WebAsKB_PtrNet_Model(self.input_lang)

        # train using training scheme
        self.net = NNRun(model, self.pairs_train, self.pairs_dev)

    def train(self):
        self.net.run_training()

    def eval(self):
        model_output = self.net.evaluate()
        with open(config.split_points_dir + config.EVALUATION_SET + '.json', 'w') as outfile:
            outfile.write(json.dumps(model_output))



