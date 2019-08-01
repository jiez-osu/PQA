import codecs
import os
import sys
import traceback
import numpy as np
from scipy.sparse import csr_matrix
import cPickle as pickle
import gc
import os
import pdb

_EMBED_DIM = int(os.environ['PQA_GLOVE_DIM'])
_GLOVE_VECTOR_PATH = os.environ['PQA_GLOVE_PATH']

import logging
logger = logging.getLogger('root')

# def load_or_make(*args, **kwargs):
#   datapath = './data-preprocessed-local'
#   filename = os.path.join(datapath, 'glove_reader.pickle')
#   if os.path.isfile(filename):
#     logger.info('Loading from {} ...'.format(filename))
#     with open(filename, 'r') as f:
#       glove_reader = pickle.load(f)
#   else:
#     logger.info('Preprocessing data...')
#     glove_reader = GloveReader(*args, **kwargs)
#     glove_reader.read()
#     logger.info('Storing into {} ...'.format(filename))
#     with open(filename, 'w') as f:
#       pickle.dump(glove_reader, f)
#   return glove_reader


class GloveReader:
  """Read GloVe output as well as raw QA data"""

  def __init__(self, 
               vector_path=_GLOVE_VECTOR_PATH, 
               vocab_path=None):
    self._vector_path = vector_path
    self._vocab_path = vocab_path
    self.embed_matrix = None
    self.embed_dim = _EMBED_DIM
    self.vocabulary = {}
    self.idx2word = {}
    self.vocabulary['<pad>'] = 0
    self.idx2word[0] = '<pad>'
    self.vocabulary['<unk>'] = 1
    self.idx2word[1] = '<unk>'
    self.vocab_size = 2

  def _read_vocab(self):
    idx = self.vocab_size
    with codecs.open(self._vocab_path, 'r', 'utf-8') as f:
      for line in f:
        word, count = line.rstrip().split(' ')
        if word == '<pad>': continue
        self.idx2word[idx] = word
        self.vocabulary[word] = idx
        idx += 1
    assert len(self.vocabulary) == len(self.idx2word)
    self.vocab_size = len(self.vocabulary)

  def _read_vector(self, 
                   read_vocab=False):
    idx = self.vocab_size
    with open(self._vector_path, 'r') as f:
      vectors = {}
      for line in f:
        vals = line.split()
        try: assert vals[0] not in vectors
        except AssertionError:
          logger.warning('Duplicate word: {}'.format(vals[0]))
          continue
        try: assert len(vals) == _EMBED_DIM + 1
        except AssertionError:
          logger.warning('Incorrect dimention: {}'.format(len(vals)))
        vectors[vals[0]] = [ float(x) for x in vals[1:] ]
        if read_vocab:
          self.idx2word[idx] = vals[0]
          self.vocabulary[vals[0]] = idx
          idx += 1
    self.vocab_size += len(vectors)
    embed_matrix = np.zeros((self.vocab_size, self.embed_dim))
    for word, vec in vectors.items():
      try:
        embed_matrix[self.vocabulary[word]] = vec
      except IndexError:
        pdb.set_trace()
    assert self.vocab_size == len(self.idx2word)
    assert self.vocab_size == len(self.vocabulary)
    self.embed_matrix = embed_matrix

  def read(self):
    if self._vocab_path is None:
      self._read_vector(read_vocab=True)
    else:
      self._read_vocab()
      self._read_vector(read_vocab=False)

  def BOW(self, words):
    ids = []
    for w in words:
      wl = w.lower()
      if wl in self.vocabulary:
        ids.append(self.vocabulary[wl])
      else:
        print wl.encode('utf-8')
    ids = np.asarray(list(set(ids)), dtype=np.int)
    n_words = len(ids)
    bow = csr_matrix((np.ones(n_words, dtype=np.int), (np.zeros(n_words, dtype=np.int), ids)), shape=(1, self.vocab_size))
    return bow


if __name__ == '__main__':
  gr = GloveReader()
  gr.read()
  pdb.set_trace()
