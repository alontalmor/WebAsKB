from config import config
from io import open
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim
import datetime
import json

# A general training and evaluation class for neural networks


class NNRun():
    def __init__(self, model, pairs_train, pairs_dev):
        self.pairs_train = pairs_train
        self.pairs_dev = pairs_dev
        self.model = model

        model.init_optimizers()

    def run_training(self):
        loss = 0
        self.max_testing_accuracy = 0
        self.start = datetime.datetime.now()

        self.train_loss = 0  # Reset every print_every
        self.stats = []
        # For early stopping
        self.best_accuracy = 0
        self.best_accuracy_iter = 0
        self.iteration = 0

        random.seed(7)
        inds = range(len(self.pairs_train))

        while self.iteration < self.best_accuracy_iter + config.NO_IMPROVEMENT_ITERS_TO_STOP:
            self.iteration += 1
            chosen_ind = random.choice(inds)

            training_pair = self.pairs_train[chosen_ind]
            input_variable = training_pair['x']
            target_variable = training_pair['y']

            # Teacher forcing
            if config.use_teacher_forcing and self.iteration < config.teacher_forcing_full_until:
                teacher_forcing = True
            elif config.use_teacher_forcing and self.iteration < config.teacher_forcing_partial_until:
                teacher_forcing = True if random.random() < 0.5 else False
            else:
                teacher_forcing = False

            train_loss, result, loss = self.model.forward(input_variable, target_variable, loss,
                                                    DO_TECHER_FORCING=teacher_forcing)

            # computing gradients
            if self.iteration % config.MINI_BATCH_SIZE == 0:
                self.train_loss += train_loss

                loss.backward()

                # OPTIMIZER STEP HERE
                self.model.optimizer_step()

                loss = 0

            if self.iteration % config.print_every == 0:
                print('--- iteration ' + str(self.iteration) +  ' run-time ' + str(datetime.datetime.now() - self.start) +  ' --------')
                print('trainset loss %.4f' % (self.train_loss / config.print_every))
                self.train_loss = 0

            if self.iteration % config.evaluate_every == 0:
                print('-- Evaluating on devset --- ')
                print('prev max adjusted accuracy %.4f' % (self.best_accuracy))
                self.evaluate()

                if self.best_accuracy + 0.001 < self.curr_accuracy:
                    self.model.save_model()
                    self.best_accuracy = self.curr_accuracy
                    self.best_accuracy_iter = self.iteration

    def evaluate(self, gen_model_output=True):
        model_output = []
        pairs_dev = [self.pairs_dev[i] for i in range(len(self.pairs_dev))]
        sample_size = min(config.max_evalset_size, len(self.pairs_dev))
        self.model.init_stats()

        self.test_loss = 0
        accuracy_avg = 0
        for test_iter in range(0, sample_size):
            testing_pair = pairs_dev[test_iter]

            test_loss , result, loss  = self.model.forward(testing_pair['x'], testing_pair['y'])
            self.test_loss += test_loss

            if gen_model_output:
                model_output += self.model.format_model_output(testing_pair, result)

            # cases in which no result or target exist will be considered a mistake (equivalent to accuracy append 0)
            if result == [] or len(testing_pair['y'])==0:
                continue

            accuracy_avg += self.model.evaluate_accuracy(testing_pair['y'], result)

        self.curr_accuracy = accuracy_avg / sample_size

        print('evalset loss %.4f' % (self.test_loss/sample_size))
        print('adjusted accuracy %.4f' % (self.curr_accuracy))

        self.model.print_stats(sample_size)

        if gen_model_output:
            return model_output
        else:
            return True





