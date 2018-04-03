
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from Models.Pytorch.encoder import EncoderRNN
from Models.Pytorch.attention_decoder import AttnDecoderRNN
from config import config

class WebAsKB_PtrNet_Model():
    def __init__(self , input_lang):

        if config.LOAD_SAVED_MODEL:
            self.encoder = torch.load(config.neural_model_dir  + 'encoder.pkl')
            self.decoder = torch.load(config.neural_model_dir  + 'decoder.pkl')
        else:
            self.encoder = EncoderRNN(input_lang.n_words, config.hidden_size)
            self.decoder = AttnDecoderRNN(config.output_size, config.hidden_size)

        self.criterion = nn.CrossEntropyLoss()
        
    def init_stats(self):
        self.avg_exact_token_match = 0
        self.exact_match = 0
        self.comp_accuracy = 0
        self.avg_one_tol_token_match = 0
        self.exact_match_one_tol = 0
        self.p1_accuracy = 0
        self.p2_accuracy = 0
        self.p1_1_right_accuracy = 0
        self.p1_1_left_accuracy = 0

    def init_optimizers(self):
        self.encoder_optimizer = optim.Adagrad(self.encoder.parameters(), lr=config.LR, lr_decay=config.ADA_GRAD_LR_DECAY,
                                          weight_decay=config.ADA_GRAD_L2)
        self.decoder_optimizer = optim.Adagrad(self.decoder.parameters(), lr=config.LR, lr_decay=config.ADA_GRAD_LR_DECAY,
                                          weight_decay=config.ADA_GRAD_L2)

    def optimizer_step(self):
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

    def evaluate_accuracy(self, target_variable, result):
        accuracy = 0
        if config.use_cuda:
            delta = [abs(target_variable.cpu().view(-1).data.numpy()[i] - result[i]) for i in range(len(result))]
        else:
            delta = [abs(target_variable.view(-1).data.numpy()[i] - result[i]) for i in range(len(result))]

        if delta[0] == 0:
            accuracy += 0.4
        if len(delta) > 1:
            if delta[1] == 0:
                accuracy += 0.3
            if delta[1] == 1:
                accuracy += 0.15
            if delta[1] == 2:
                accuracy += 0.05
            if delta[2] == 0:
                accuracy += 0.3
            if delta[2] == 1:
                accuracy += 0.15
            if delta[2] == 2:
                accuracy += 0.05

        abs_delta_array = np.abs(np.array(delta))
        self.avg_exact_token_match += np.mean((abs_delta_array == 0) * 1.0)
        self.avg_one_tol_token_match += ((abs_delta_array[0] == 0) * 1.0 + np.sum((abs_delta_array[1:] <= 1) * 1.0)) / 3.0
        self.exact_match += (np.mean((abs_delta_array == 0) * 1.0) == 1.0) * 1.0
        self.exact_match_one_tol += ((abs_delta_array[0] == 0) & (np.mean((abs_delta_array <= 1) * 1.0) == 1.0)) * 1.0

        if config.use_cuda:
            target = target_variable.cpu().view(-1).data.numpy()
        else:
            target = target_variable.view(-1).data.numpy()
        if target[0] == result[0]:
            self.comp_accuracy += 1
        if len(delta) > 1:
            if target[1] == result[1]:
                self.p1_accuracy += 1
            if target[1] == result[1] - 1:
                self.p1_1_right_accuracy += 1
            if target[1] == result[1] + 1:
                self.p1_1_left_accuracy += 1
            if target[2] == result[2]:
                self.p2_accuracy += 1

        return accuracy

    def print_stats(self, sample_size):

        comp_accuracy_avg = self.comp_accuracy / sample_size
        p1_accuracy_avg = self.p1_accuracy / sample_size
        p2_accuracy_avg = self.p2_accuracy / sample_size
        p1_1_right_accuracy_avg = self.p1_1_right_accuracy / sample_size
        p1_1_left_accuracy_avg = self.p1_1_left_accuracy / sample_size

        print('avg_exact_token_match %.4f' % (self.avg_exact_token_match / sample_size))
        print('exact_match %.4f' % (self.exact_match / sample_size))
        print('avg_one_tol_token_match %.4f' % (self.avg_one_tol_token_match / sample_size))
        print('exact_match_one_tol %.4f' % (self.exact_match_one_tol / sample_size))

        print('comp_accuracy %.4f' % (comp_accuracy_avg))
        print('p1_accuracy %.4f' % (p1_accuracy_avg))
        print('p2_accuracy %.4f' % (p2_accuracy_avg))
        print('p1_1_right_accuracy %.4f' % (p1_1_right_accuracy_avg))
        print('p1_1_left_accuracy %.4f' % (p1_1_left_accuracy_avg))

    def format_model_output(self,pairs_dev, result):
        input_tokens = [token['dependentGloss'] for token in pairs_dev['aux_data']['sorted_annotations']]

        output_sup = pairs_dev['y'].view(-1).data.numpy()

        comp_names = ['composition', 'conjunction']
        comp = comp_names[int(result[0]) - 1]
        if len(output_sup) > 0:
            comp_sup = comp_names[int(output_sup[0]) - 1]
        else:
            comp_sup = ''
        p1 = int(result[1]) - 3
        if len(output_sup) > 0:
            p1_sup = int(output_sup[1]) - 3
        else:
            p1_sup = ''

        p2 = int(result[2]) - 3

        p2_sup = None
        if len(output_sup) > 0:
            p2_sup = int(output_sup[2]) - 3
        else:
            p2_sup = ''

        question_tokens = input_tokens

        if comp == 'conjunction':
            split_part1 = question_tokens[0:p1 + 1]
            split_part2 = question_tokens[p1 + 1:]

            split_part1 = ' '.join(split_part1).replace(" 's", "'s")
            split_part2 = ' '.join(split_part2).replace(" 's", "'s")

            if p2 != 0 and p2 < len(question_tokens):
                split_part2 = question_tokens[p2] + ' ' + split_part2

        elif comp == 'composition':
            split_part1 = ''
            split_part2 = ''
            if p2 is not None:
                split_part1 = ' '.join(question_tokens[p1:p2 + 1]).replace(" 's", "'s")
                split_part2 = ''
                if p1 > 0:
                    split_part2 += ' '.join(question_tokens[0:p1]).replace(" 's", "'s")
                split_part2 += ' %composition '
                if p2 + 1 < len(question_tokens):
                    split_part2 += ' '.join(question_tokens[p2 + 1:]).replace(" 's", "'s")
                split_part2 = split_part2.strip()
        else:
            split_part1 = ''
            split_part2 = ''

        return [{'ID': pairs_dev['aux_data']['ID'], 'comp': comp, 'comp_sup': comp_sup,
                           'same_comp': int(comp == comp_sup), 'p1': p1, 'p1_sup': p1_sup, 'p2': p2, \
                           'p2_sup': p2_sup, 'split_part1': split_part1, \
                           'split_part2': split_part2,
                           'question': pairs_dev['aux_data']['question'], \
                           'answers': pairs_dev['aux_data']['answers']}]

    def save_model(self):
        torch.save(self.encoder, config.neural_model_dir + 'encoder.pkl')
        torch.save(self.decoder, config.neural_model_dir + 'decoder.pkl')

    def forward(self,input_variable, target_variable, loss=0, DO_TECHER_FORCING=False):
        encoder_hidden = self.encoder.initHidden()

        input_length = len(input_variable)
        target_length = len(target_variable)

        encoder_outputs = Variable(torch.zeros(config.MAX_LENGTH, self.encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if config.use_cuda else encoder_outputs

        encoder_hiddens = Variable(torch.zeros(config.MAX_LENGTH, self.encoder.hidden_size))
        encoder_hiddens = encoder_hiddens.cuda() if config.use_cuda else encoder_hiddens

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]
            encoder_hiddens[ei] = encoder_hidden[0][0]

        decoder_input = Variable(torch.LongTensor([[config.SOS_token]]))
        decoder_input = decoder_input.cuda() if config.use_cuda else decoder_input

        decoder_hidden = encoder_hidden
        result = []
        # Without teacher forcing: use its own predictions as the next input
        sub_optimal_chosen = False
        for di in range(3):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_hidden, encoder_hiddens, encoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            if DO_TECHER_FORCING:
                decoder_input = target_variable[di]
            else:
                decoder_input = Variable(torch.LongTensor([[int(np.argmax(decoder_attention.data[0].tolist()))]]))

            # we are computing logistical regression vs the hidden layer!!
            if len(target_variable)>0:
                loss += self.criterion(decoder_attention, target_variable[di])

            result.append(np.argmax(decoder_attention.data[0].tolist()))

        if type(loss)!=int:
            loss_value = loss.data[0] / target_length
        else:
            loss_value = 0
        return loss_value , result, loss