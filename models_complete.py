import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        '''
        :param input: An input sequence vector. (Dims ==> [Seqence_length X 1])
        :param hidden: An hidden vector. (Dims ==> [1 X 1 X hidden_size])
        :return output: An output vector from RNN module (Dims ==> [1 X 1 X hidden_size])
        :return hidden: An hidden vector from RNN module (Dims ==> [1 X 1 X hidden_size])
        '''
        '''
        # Things you need to implement
         - Setup the those following parts.
           1. Embedding Layer for input layer
           2. RNN Layer which has an option to set multiple layers
        '''
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        '''
        :param input: An input sequence vector. (Dims ==> [1 X Sequence_length])
        :param hidden: An hidden vector. (Dims ==> [1 X 1 X hidden_size])
        :param encoder_outputs: A sequence of hidden vectors from the encoder. (Dims ==> [EncSeq_len X hidden_size]) 
        :return output: A prediction vector from RNN Module (Dims ==> [1 X output_size]
        :return hidden: A hidden vector from RNN Module (Dims ==> [1 X 1 X hidden_size]
        :return attn_weights: An attention weight vector for one step (Dims ==> [1 X Sequence_length])
        '''
        '''
        # Things you need to implement
          - Setup the those following parts.
            1. Embedding Layer for input vector
            2. Attention Module to apply RNN (or LSTM, GRU) module
            3. RNN Layer which has an option to set multiple layers
            4. Output Layer to get predictions from inputs         
        '''
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result
