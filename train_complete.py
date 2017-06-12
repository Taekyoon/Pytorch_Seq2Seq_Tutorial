import random
import time

import torch
from torch import optim
from torch import nn
from torch.autograd import Variable

from train_util import variables_from_pair, time_since

teacher_forcing_ratio = 0.5
SOS_token = 0
EOS_token = 1


def train(input_variable, target_variable, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, max_length):
    '''
    :param input_variable: An input sequence to inject encoder (Dims ==> [1 X sequence_length])
    :param target_variable: An target sequence to inject decoder (Dims ==> [1 X sequence_length])
    :param encoder: An encoder model
    :param decoder: An decoder model
    :param encoder_optimizer: encoder_optimizer function
    :param decoder_optimizer: decoder_optimizer function
    :param criterion: criterion function
    :param max_length: sequence maximum length for both encoder and decoder modules
    :return: An average loss value of predictions from each steps
    '''
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        '''
        ## Teacher forcing: Feed the target as the next input
        # Things you need to implement
         1. Get the prediction vector from decoder
         2. Accumulate loss value using 'criterion'
         3. Assign a ground truth letter to 'decoder_input'
        '''
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output[0], target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        '''
        ## Without teacher forcing: use its own predictions as the next input
        # Things you need to implement
         1. Get the prediction vector from decoder
         2. Get 'topk'(or argmax) value from prediction vector
         3. Assign a predicted value to 'decoer_input' using 'Variable'
         4. Accumulate loss value using 'criterion'
         5. Set a condition whether a predicted letter is EOS_token 
            (a variable 'EOS_token' has already assigned in this module)
        '''
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))

            loss += criterion(decoder_output[0], target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def train_iters(encoder, decoder, input_lang, output_lang, pairs, n_iters, max_len,
                print_every=100, plot_every=10, learning_rate=0.01):
    '''
    :param encoder: An Encoder model
    :param decoder: An Decoder model
    :param input_lang: Size of input sequence symbols
    :param output_lang: Size of output sequence symbols
    :param pairs: Training tuples which have input sequences and output sequences
    :param n_iters: Number of iterations per a step
    :param max_len: Maximum length of sequence for each models
    :param print_every: Print loss value every this param step
    :param plot_every: Get loss value to plot a graph every this param step
    :param learning_rate: A hyperparameter to tune a learning rate 
    :return: A loss value list to plot a graph
    '''
    global EOS_token
    global SOS_token
    EOS_token = input_lang.word2index["EOS"]
    SOS_token = input_lang.word2index["SOS"]

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    '''
    # Things you need to implement
     1. Declare your optimizer functions for each models
     2. Trim learning rate to train model
     3. Optimizer functions are from 'torch.optim' Module  
    '''
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = []

    for _ in range(n_iters):
        training_pairs.append(variables_from_pair(input_lang, output_lang, random.choice(pairs), max_len))

    '''
    # Things you need to implement
     1. Declare your criterion
     2. Criterion functions are from 'torch.nn' Module
    '''
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, max_len)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    return plot_losses
