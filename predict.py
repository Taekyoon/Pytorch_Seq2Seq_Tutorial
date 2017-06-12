import random

import torch
from torch.autograd import Variable

from train_util import variable_from_sentence


class ModelPredictor(object):

    def __init__(self, encoder, decoder, input_lang, output_lang, max_length):
        self.encoder = encoder
        self.decoder = decoder
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.max_length = max_length

    def evaluate(self, sentence):
        SOS_token = self.input_lang.word2index["SOS"]
        EOS_token = self.input_lang.word2index["EOS"]

        input_variable = variable_from_sentence(self.input_lang, sentence, self.max_length)
        input_length = input_variable.size()[0]
        encoder_hidden = self.encoder.init_hidden()

        encoder_outputs = Variable(torch.zeros(self.max_length, self.encoder.hidden_size))

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_variable[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(self.max_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == EOS_token:
                break
            else:
                decoded_words.append(self.output_lang.index2word[ni])

            decoder_input = Variable(torch.LongTensor([[ni]]))

        return decoded_words

    def evaluate_randomly(self, pairs, n=10):
        match = 0
        for i in range(n):
            pair = random.choice(pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words = self.evaluate(pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')
            if pair[1] == output_sentence:
                match += 1
        print("accuracy: ", (match / n) * 100, "%")

    def predict_sentence(self, sentence):
        return ' '.join(self.evaluate(sentence))
