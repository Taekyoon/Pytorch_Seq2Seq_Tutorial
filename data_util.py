import glob
import unicodedata
import re
from random import shuffle

'''
Only use this function in this module below

prepare_data(lang1_name, lang2_name, n_words, reverse=False)
'''


def read_sentences(input_filename, target_filename, n_words, down_margin):
    input_lines = open(input_filename).read().strip().split('\n')
    target_lines = open(target_filename).read().strip().split('\n')
    
    targets = []
    for i, input_line in enumerate(input_lines):
        seq_list_len = len(input_line.split(' '))
        if n_words - down_margin <= seq_list_len < n_words:
            targets.append((input_line, target_lines[i]))

    return targets


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_langs(lang1, lang2, n_words, down_margin, reverse=False):
    all_filenames = glob.glob('data/europarl-v*.fr-en.*')
    print(all_filenames)

    print("Reading lines...")

    pairs = read_sentences(all_filenames[0], all_filenames[1], n_words, down_margin)
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filter_pair(pair, func):
    return func(pair[0]), func(pair[1])


def filter_pairs(pairs, func):
    return [(filter_pair(pair, func)) for pair in pairs]


def split_data(data, test_ratio):
    test_set_size = int(len(data) * test_ratio)
    return data[:test_set_size], data[test_set_size:]


def prepare_data(lang1_name, lang2_name, n_words, down_margin, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, n_words, down_margin, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs, normalize_string)
    ret_pairs = []
    max_input_len = 0
    max_target_len = 0

    for pair in pairs:
        input_seq_len = len(pair[0].split(" "))
        target_seq_len = len(pair[0].split(" "))
        if n_words - down_margin <= input_seq_len < n_words and \
           n_words - down_margin <= target_seq_len < n_words:
            ret_pairs.append(pair)
        
            if max_input_len < input_seq_len:
                max_input_len = input_seq_len
            if max_target_len < target_seq_len:
                max_target_len = target_seq_len

    print("Trimmed to %s sentence pairs" % len(ret_pairs))

    print("Indexing words...")
    for pair in ret_pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    print("Spliting sentence pairs...")
    test_pairs, train_pairs = split_data(ret_pairs, 0.2)

    print("====== Total Data ======")
    print("Train Sentence pairs: ", len(train_pairs))
    print("Test Sentence pairs: ", len(test_pairs))
    print(lang1_name, 'n_words: ', input_lang.n_words, 'max_len: ', max_input_len)
    print(lang2_name, 'n_words: ', output_lang.n_words, 'max_len: ', max_target_len)

    return input_lang, output_lang, train_pairs, test_pairs


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1, "PAD": 2, "UNK": 3}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"PAD", 3:"UNK"}
        self.n_words = 4  # Count SOS and EOS

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
