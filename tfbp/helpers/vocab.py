import json
import numpy as np
from tfbp.helpers.definitions import vocab_path


# Load JSON array of vocab words from file
with open(vocab_path) as f:
  vocab = json.load(f)

vocab_size = len(vocab)

# Pad_char should be last character in vocab list
pad_char = vocab[-1]

word_vectors = np.zeros([vocab_size, vocab_size])
word2index = {}

i = 0
for word in vocab:
  word2index[word] = i
  word_vectors[i][i] = 1
  i += 1

index2word = {v: k for k, v in word2index.iteritems()}


def word2vec(word):
  return word_vectors[word2index.get(word)]


def vec2word(vec):
  index = list(vec).index(1.0)
  return index2word[index]


def label2vec(label):
  return [list(word2vec(c)) for c in label]


def vec2label(vec):
  return ''.join([vec2word(v) for v in vec]).rstrip(pad_char)