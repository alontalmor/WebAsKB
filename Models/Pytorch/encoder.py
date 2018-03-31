import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as weight_init
from config import config

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, config.EMBEDDING_VEC_SIZE)
        self.GRU = nn.GRU(config.EMBEDDING_VEC_SIZE * 2 , hidden_size)

    def forward(self, input, hidden):
        if len(input[0]) == 1:
            try:
                word_token_embedded = self.embedding(input[0]).view(1, 1, -1)
            except:
                word_token_embedded = Variable(torch.zeros(50).view(1, 1, -1))
        else:
            word_token_embedded = input[0].view(1, 1, -1)

        if len(input[1]) == 1:
            dep_parse_embedded = self.embedding(input[1]).view(1, 1, -1)
        else:
            dep_parse_embedded = input[1].view(1, 1, -1)

        #if len(input[2]) == 1:
        #    dependant_word_embedded = self.embedding(input[2]).view(1, 1, -1)
        #else:
        #    dependant_word_embedded = input[2].view(1, 1, -1)

        output = torch.cat((word_token_embedded[0], dep_parse_embedded[0]), 1).view(1, 1,-1)
        #output = torch.cat((word_token_embedded[0], dep_parse_embedded[0],dependant_word_embedded[0]), 1).view(1, 1, -1)
        for i in range(self.n_layers):
            output, hidden = self.GRU(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(weight_init.xavier_normal(torch.Tensor(1, 1 , self.hidden_size)))
        #result = Variable(torch.zeros(1, 1, self.hidden_size))
        if config.use_cuda:
            return result.cuda()
        else:
            return result
