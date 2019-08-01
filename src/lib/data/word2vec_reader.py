import codecs
import os
import sys
import traceback
import numpy as np
from scipy.sparse import csr_matrix
import cPickle as pickle
import gc
import pdb
_EMBED_DIM = 200
_WORD2VEC_IN_VECTOR_PATH = './../models-master/tutorials/embedding/temp/embed_in_mat.pickle'
_WORD2VEC_OUT_VECTOR_PATH = './../models-master/tutorials/embedding/temp/embed_out_mat.pickle' 
_WORD2VEC_VOCAB_PATH = './../models-master/tutorials/embedding/temp/vocab.txt' 

import logging
logger = logging.getLogger('root')


class Word2vecReader:

  def __init__(self, 
               in_vector_path=_WORD2VEC_IN_VECTOR_PATH, 
               out_vector_path=_WORD2VEC_OUT_VECTOR_PATH, 
               vocab_path=_WORD2VEC_VOCAB_PATH):
    self._in_vector_path = in_vector_path
    self._out_vector_path = out_vector_path
    self._vocab_path = vocab_path
    self.in_embed_matrix = None
    self.out_embed_matrix = None
    self.embed_dim = 0
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
        if word == 'UNK': continue
        self.idx2word[idx] = word
        self.vocabulary[word] = idx
        idx += 1
    assert len(self.vocabulary) == len(self.idx2word)
    self.vocab_size = len(self.vocabulary)

  def _read_vector(self, 
                   in_embed_path=_WORD2VEC_IN_VECTOR_PATH ,
                   out_embed_path=_WORD2VEC_OUT_VECTOR_PATH):
    with open(in_embed_path, 'r') as f:
      in_embed_matrix = pickle.load(f)
      self.in_embed_matrix = in_embed_matrix[1 :]
    with open(out_embed_path, 'r') as f:
      out_embed_matrix = pickle.load(f)
      self.out_embed_matrix = out_embed_matrix[1 :]
    assert(self.in_embed_matrix.shape == self.out_embed_matrix.shape)
    assert(self.in_embed_matrix.shape[0] == self.vocab_size - 2)
    self.embed_dim = self.in_embed_matrix.shape[1]
    self.in_embed_matrix = np.concatenate([np.zeros([2, self.embed_dim]), self.in_embed_matrix])
    self.out_embed_matrix = np.concatenate([np.zeros([2, self.embed_dim]), self.out_embed_matrix])
    self.embed_matrix = self.in_embed_matrix

  def read(self):
    self._read_vocab()
    self._read_vector()

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
  gr = Word2vecReader()
  gr.read()
  pdb.set_trace()
