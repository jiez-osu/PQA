# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import traceback
import os
import spacy
import codecs
import json
from collections import defaultdict, Counter
import glove_reader
import word2vec_reader
# import re
import numpy as np
from scipy.sparse import csr_matrix
import gc
import dill
import pdb
from sklearn import preprocessing
from review import Reviews
from qa import QA
from dataset_utils import token, correct_token, correct_paragraph

# _DATA_PATH = '/home/jzhao/amazon-qa/data/baby_qa_review.txt'
# _DATA_PATH = '../../data_for_my_product_qa/baby_qa_review.txt'
_DATA_PATH = '../../data_for_my_product_qa/electronics_qa_review.txt'

MIN_TOKEN_FREQ = 5
PAD_WID = 0
UNK_WID = 1
# KEEP_ANS_MIN_WORD = 2 

import logging
logger = logging.getLogger('root')


class CorpusReader:

  def __init__(self, 
               datapath=_DATA_PATH, 
               maxline=-1,
               num_reviews=20,
               if_only_top_ans=True,
               if_print=False,
               if_make_vocab=True,
               load_glove=True):
    self._datapath = datapath
    self._maxline = maxline
    self.num_reviews = num_reviews
    self.if_only_top_ans = if_only_top_ans

    self.nlp = spacy.load('en')
    self.nlp.vocab[u"'s"].is_stop = True
    self.nlp.vocab[u"'m"].is_stop = True
    self.nlp.vocab[u"'re"].is_stop = True
    self.nlp.vocab[u"n't"].is_stop = True
    self.nlp.vocab[u"hasn't"].is_stop = True
    self.nlp.vocab[u"haven't"].is_stop = True
    self.vocab_cnt = defaultdict(int)
    # Reconciled vocabulary
    self.word2id = {"<pad>" : PAD_WID,
                    "<unk>" : UNK_WID}  # word to index
    self.id2word = {PAD_WID : "<pad>",
                    UNK_WID : "<unk>"}  # index to word
    self.id2freq = {PAD_WID : -1,
                    UNK_WID : -1}  # index to word frequency in the corpus
    self.vocab_size = 2
    self.embed_matrix = None
    self.cntxt_embed_matrix = None
    self.w_embed_size = None
    # POS
    self.pos_cnt = defaultdict(int)
    self.pos2id = {"<pad>": 0}
    self.id2pos = {0: "<pad>"}
    self.num_pos_tags = 1
    # Need to go through data once to get vocabulary
    if if_make_vocab:
      logger.info("Making vocabulary ...")
      self._make_vocab(if_print=if_print, load_glove=load_glove)
  
  def _make_vocab(self, if_print=False, load_glove=True):
    logger.info("Read pretrained GloVe ...")
    
    gr = glove_reader.GloveReader()
    if load_glove:
      gr.read() # NOTE: Disabled to save time debugging.
    # gr = word2vec_reader.Word2vecReader()
    # gr.read()

    logger.info("Go through corpus to make vocabulary ...")
    for review, qa_list, _ in self.data_iterator(_find_words=False):
      for qa in qa_list:
        assert qa.__class__.__name__ == 'QA'
        self.count_vocab(qa.q_doc)
        for a_doc in qa.a_docs:
          self.count_vocab(a_doc)

      # assert review.__class__.__name__ == 'Reviews'
      # for r_doc in review.r_docs:
      #   self.count_vocab(r_doc)

    logger.info("Reconcile with GloVe vocab ...")
    token_list = sorted(self.vocab_cnt, key=self.vocab_cnt.get, reverse=True)
    word_embeddings, cntxt_word_embeddings = [], []
    oov_tokens = []
    rare_tokens = []
    idx = len(self.word2id)
    for token in token_list:
      if token == "<pad>" or token == "<unk>":
        pdb.set_trace()
      token_freq = self.vocab_cnt[token]
      if if_print: print(u"{0:<20} : {1}".format(token, token_freq))
      if token in gr.vocabulary:
        word_embeddings.append(gr.embed_matrix[gr.vocabulary[token]])
        self.word2id[token] = idx
        self.id2word[idx] = token
        self.id2freq[idx] = token_freq
        idx += 1
      elif token_freq >= MIN_TOKEN_FREQ:
        oov_tokens.append(token)
        word_embeddings.append(np.random.uniform(-0.01, 0.01, gr.embed_dim)) # NOTE: NEED DOWN SCALING
        self.word2id[token] = idx
        self.id2word[idx] = token
        self.id2freq[idx] = token_freq
        idx += 1
      else:
        rare_tokens.append(token)
    assert len(self.word2id) == idx
    assert len(self.id2word) == idx
    assert len(self.id2freq) == idx
    if if_print:
      print(oov_tokens)
      print(rare_tokens)
    logger.info("OOV tokens (out of GloVe vocab): {}".format(len(oov_tokens)))
    logger.info("Rare tokens (too rare, out of our vocab, <UNK> tokens): {}".format(len(rare_tokens)))
    logger.info("Vocabulary size: {}".format(idx))
    self.vocab_size = idx

    # Finalize word embedding
    self.w_embed_size = gr.embed_dim
    self.embed_matrix = np.concatenate([np.zeros([2, gr.embed_dim]),
                                        preprocessing.normalize(np.stack(word_embeddings), axis=1)])
    assert (self.vocab_size, self.w_embed_size) == self.embed_matrix.shape
    del gr
    gc.collect()

    # POS
    pos_list = sorted(self.pos_cnt, key=self.pos_cnt.get, reverse=True)
    pos_display = ""
    for i, pos in enumerate(pos_list):
      pos_freq = self.pos_cnt[pos]
      pos_display += u"\n{0:<10} : {1}".format(pos, pos_freq)
      self.pos2id[pos] = i + 1
      self.id2pos[i + 1] = pos
    assert(len(self.pos2id) == len(self.id2pos))
    self.num_pos_tags = len(self.pos2id)
    logger.info(u"POS tags: " + pos_display)
    logger.info(u"Number of POS tags: {}".format(self.num_pos_tags))

  def count_vocab(self, doc):
    for w in doc:
      t = correct_token(token(w))
      if t:
        self.vocab_cnt[t] += 1
        self.pos_cnt[w.pos_] += 1
  
  def token_to_word_id(self, words, exclude_unk=True, exclude_pad=True):
    """From string to int"""
    wids = []
    for word in words:
      w = word
      if w in self.word2id:
        wid = self.word2id[w]
        if not (wid == PAD_WID and exclude_pad):
          wids.append(wid)
      else:
        if not exclude_unk: wids.append(UNK_WID)
        try:
          logger.debug(u"Unknown word: {}".format(w.encode('utf-8')))
        except UnicodeError:
          excinfo = sys.exc_info()
          traceback.print_exception(*excinfo)
    return wids

  def word_id_to_bow(self, wids, binary=False, dense=False):
    """From word ids to BOW vectors."""
    if binary: # Only consider whether appear or not
      ids = np.asarray(list(set(wids)), dtype=np.int)
      n_words = len(ids)
      bow = csr_matrix((np.ones(n_words, dtype=np.int),
                        (np.zeros(n_words, dtype=np.int), ids)),
                       shape=(1, self.vocab_size))
    else: # Count actual appearance time
      id_cnt = Counter(wids)
      n_words = len(id_cnt)
      bow = csr_matrix((np.asarray(id_cnt.values()),
                       (np.zeros(n_words, dtype=np.int), id_cnt.keys())), 
                       shape=(1, self.vocab_size)) 
    if dense: bow = bow.toarray()
    return bow

  def _read_question(self, qa):
    question = correct_paragraph(qa['question'])
    return question

  def _read_answers(self, qa):
    answers = []
    for answer in qa['answers']:
      clean_answer = correct_paragraph(answer['answer'])  
      answers.append(clean_answer)
      if self.if_only_top_ans: 
        break
    return answers
  
  def _read_reviews(self, reviews):
    clean_reviews = []
    for review in reviews:
      clean_review = correct_paragraph(review["reviewText"])
      clean_reviews.append(clean_review)
    return clean_reviews

  def data_iterator(self, _read_reviews=False, _find_words=False):
    cnt = 0
    with codecs.open(self._datapath, 'r', 'utf-8') as f:
      for line in f:
        if not line: continue
        try: prod = json.loads(line)
        except ValueError:
          excinfo = sys.exc_info()
          traceback.print_exception(excinfo[0], excinfo[1], excinfo[2])
          pdb.set_trace()
        if prod['reviews'] is None:
          continue
        cnt += 1
        if 0 <= self._maxline <= cnt:
          break

        # Preprocess reviews
        review = None
        if _read_reviews:
          reviews = self._read_reviews(prod['reviews'])
          review = Reviews(self.nlp, prod['asin'], reviews, sent_split=True)
          if len(review.r_docs) < self.num_reviews:
            continue  # Filter out products with insufficient number of reviews

        # Preprocess question-answer
        qa_list = []
        review_focus_mask = []
        for qid, qa in enumerate(prod['qa']):
          question = self._read_question(qa)
          answers = self._read_answers(qa)
          qa = QA(self.nlp, prod["asin"], qid, question, answers)
          
          if len(qa.a_docs) == 0:
            continue
          qa_list.append(qa)
          
          if _find_words:
            # Extract question focus words
            qa.extract_words()
            # Match QA with reviews to find out which words among reviews should be focused on
            review_focus_mask.append(review.extract_words(qa.q_token_set, qa.a_token_set))

        yield(review, qa_list, review_focus_mask)






if __name__ == '__main__':
  reader = CorpusReader(maxline=200,
                        if_make_vocab=False,
                        if_print=True)
  reader.prep_word2vec_input()

  # reader.prep_GloVe_input()
  # reader.prep_MoQA_input()

  # for qa in reader.data_iterator():
  #   qa.print_question()
  #   qa.print_answers()
  #   pdb.set_trace()


