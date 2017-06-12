import time
import math
import torch
from torch.autograd import Variable


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variable_from_sentence(lang, sentence, max_len):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(lang.word2index["EOS"])

    for _ in range(max_len-len(indexes)):
        indexes.append(lang.word2index["PAD"])
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    return result


def variables_from_pair(input_lang, output_lang, pair, max_len):
    input_variable = variable_from_sentence(input_lang, pair[0], max_len)
    target_variable = variable_from_sentence(output_lang, pair[1], max_len)
    return (input_variable, target_variable)
