import sys
import traceback
import numpy as np
import scipy
from collections import defaultdict
import os
import cPickle as pickle
from scipy.sparse import csr_matrix, vstack
import gc
import pdb
import glove_reader
from dataset_utils import correct_token, token
import codecs
from collections import Counter
import json
from sklearn import preprocessing

from multiprocessing import Pool
from tqdm import tqdm

# import warnings
# warnings.filterwarnings('error')

PAD_WID = 0
UNK_WID = 1
MIN_TOKEN_FREQ = 5
MAX_VOCAB_SIZE = 100000

import logging

logger = logging.getLogger('root')

SELF_TEST_MAX_LINE = 20
MAKE_REVIEW_DATA = False
MAKE_NONE_ANSWER_DATA = True
# domain = 'Baby'
# domain = 'Appliances'
# domain = 'Patio_Lawn_and_Garden'
# domain = 'Tools_and_Home_Improvement'
# domain = 'Electronics'
domain = os.environ['PQA_DOMAIN']

# DATA_PATH = './../../word-interactions/data-preprocessed-local'
# DATA_PATH = '../data-preprocessed-local/Baby'  # New dataset, same as what Moqa used
# _DATA_SOURCE = '../../data_for_my_product_qa/%s_qa_review_spacy_preprocessed.txt' % domain.lower()
# _DATA_PATH = '../data-preprocessed-local/%s' % domain  # New dataset, same as what Moqa used
_DATA_SOURCE = os.environ['PQA_DATA_SOURCE']
_DATA_PATH = os.environ['PQA_DATA_PATH']
if not os.path.exists(_DATA_PATH):
  os.makedirs(_DATA_PATH)


class Dataset():

  def __init__(self,
               max_len=50,  # Maximum sentence length, same for questions, answers and reviews
               num_reviews=20,  # Number of review candidates for each QA pair
               selftest=False,
               if_only_top_ans=True,
               top_score_recorder=None,
               load_meta=True,
               load_vocab=True,
               load_qa=True,
               load_review=True,
               load_word_embedding=True):
    try:
      # pdb.set_trace()
      self.selftest = selftest
      if load_meta:
        self._load_meta()
      if load_vocab:
        self._load_vocab()
      if load_qa:
        self._load_qa()
      if load_review:
        self._load_review()
      if load_word_embedding:
        self._load_word_embedding()
    except IOError:
      logger.info('Stored data not found, preprocessing ...')
      self.selftest = selftest
      self.max_len = max_len
      self.num_reviews = num_reviews

      logger.info('Create vocabulary ...')
      self._create_vocab(maxline=SELF_TEST_MAX_LINE if selftest else -1,
                         load_glove=False if selftest else True)

      logger.info('Read corpus data and convert to arrays ...')
      data, review_data, asin2id = self._read_into_arrays(maxline=SELF_TEST_MAX_LINE if selftest else -1)
      self.review_data = review_data
      del review_data
      gc.collect()

      logger.info('Calculate review IDF ...')
      self.review_idf = self._get_review_idf()

      logger.info('Splitting data into train, dev, test sets ...')
      self._train_idx, self._dev_idx, self._test_idx = [], [], []
      self._train_size, self._dev_size, self._test_size = 0, 0, 0
      self._data_split(data)
      del data
      gc.collect()

      self._save_meta()
      self._save_vocab()
      self._save_qa()
      self._save_review()
      self._save_word_embedding()

    self._block_to_dense()
    self.top_score_recorder = top_score_recorder
    if self.top_score_recorder is not None:
      logger.info("Train with Pseudo Relevance Feedbacks")
    self._print_info()

  def _save_meta(self):
    if not os.path.exists(_DATA_PATH):
      os.makedirs(_DATA_PATH)
    filename = os.path.join(_DATA_PATH, 'meta{}.pickle'.format(('-selftest' if self.selftest else '')))
    logger.info('Storing meta data into {}...'.format(filename))
    with open(filename, 'wb') as f:
      meta = {'max_len': self.max_len,
              'num_reviews': self.num_reviews,
              'train_size': self._train_size,
              'dev_size': self._dev_size,
              'test_size': self._test_size}
      pickle.dump(meta, f)

  def _load_meta(self):
    filename = os.path.join(_DATA_PATH, 'meta{}.pickle'.format(('-selftest' if self.selftest else '')))
    logger.info('Loading meta data from {}...'.format(filename))
    with open(filename, 'rb') as f:
      meta = pickle.load(f)
      self.max_len = meta['max_len']
      self.num_reviews = meta['num_reviews']
      self._train_size = meta['train_size']
      self._dev_size = meta['dev_size']
      self._test_size = meta['test_size']

  def _save_vocab(self):
    if not os.path.exists(_DATA_PATH):
      os.makedirs(_DATA_PATH)
    filename = os.path.join(_DATA_PATH, 'vocabulary{}.pickle'.format(('-selftest' if self.selftest else '')))
    logger.info('Storing vocabulary into {}...'.format(filename))
    with open(filename, 'wb') as f:
      vocabulary = {'id2word': self.id2word,
                    'word2id': self.word2id,
                    # 'id2pos': self.id2pos,
                    # 'pos2id': self.pos2id,
                    'id2freq': self.id2freq,
                    'vocab_size': self.vocab_size,
                    'num_pos_tags': self.num_pos_tags,
                    'w_embed_size': self.w_embed_size}
      pickle.dump(vocabulary, f)

  def _load_vocab(self):
    filename = os.path.join(_DATA_PATH, 'vocabulary{}.pickle'.format(('-selftest' if self.selftest else '')))
    logger.info('Loading vocabulary from {}...'.format(filename))
    with open(filename, 'rb') as f:
      vocabulary = pickle.load(f)
      self.id2word = vocabulary['id2word']
      self.word2id = vocabulary['word2id']
      # self.id2pos = vocabulary['id2pos']
      # self.pos2id = vocabulary['pos2id']
      self.id2freq = vocabulary['id2freq']
      self.vocab_size = vocabulary['vocab_size']
      self.num_pos_tags = vocabulary['num_pos_tags']
      self.w_embed_size = vocabulary['w_embed_size']

  def _save_qa(self):
    if not os.path.exists(_DATA_PATH):
      os.makedirs(_DATA_PATH)
    filename = os.path.join(_DATA_PATH, 'qa{}.pickle'.format(('-selftest' if self.selftest else '')))
    logger.info('Storing question & answer data into {}...'.format(filename))
    with open(filename, 'wb') as f:
      qa_data = {'train_data': self.train_data,
                 'dev_data': self.dev_data,
                 'test_data': self.test_data}
      pickle.dump(qa_data, f)

  def _load_qa(self):
    filename = os.path.join(_DATA_PATH, 'qa{}.pickle'.format(('-selftest' if self.selftest else '')))
    logger.info('Loading question & answer data from {}...'.format(filename))
    with open(filename, 'rb') as f:
      qa_data = pickle.load(f)
      self.train_data = qa_data['train_data']
      self.dev_data = qa_data['dev_data']
      self.test_data = qa_data['test_data']

  def _save_review(self):
    if not os.path.exists(_DATA_PATH):
      os.makedirs(_DATA_PATH)
    filename = os.path.join(_DATA_PATH, 'review{}.pickle'.format(('-selftest' if self.selftest else '')))
    logger.info('Storing review data into {}...'.format(filename))
    with open(filename, 'wb') as f:
      review_data = self.review_data
      pickle.dump(review_data, f)

  def _load_review(self):
    filename = os.path.join(_DATA_PATH, 'review{}.pickle'.format(('-selftest' if self.selftest else '')))
    logger.info('Loading review data from {}...'.format(filename))
    with open(filename, 'rb') as f:
      review_data = pickle.load(f)
      self.review_data = review_data

  def _save_word_embedding(self):
    if not os.path.exists(_DATA_PATH):
      os.makedirs(_DATA_PATH)
    filename = os.path.join(_DATA_PATH, 'word_embedding{}.pickle'.format(('-selftest' if self.selftest else '')))
    logger.info('Storing word embedding data into {}...'.format(filename))
    with open(filename, 'wb') as f:
      word_embedding_data = self.embed_matrix
      pickle.dump(word_embedding_data, f)

  def _load_word_embedding(self):
    filename = os.path.join(_DATA_PATH, 'word_embedding{}.pickle'.format(('-selftest' if self.selftest else '')))
    logger.info('Loading word embedding data from {}...'.format(filename))
    with open(filename, 'rb') as f:
      word_embedding_data = pickle.load(f)
      self.embed_matrix = word_embedding_data

  def _print_info(self):
    """
    Print statistics of the dataset
    """
    logger.info('Vocabulary size : {}'.format(self.vocab_size))
    logger.info('Train question size : {}'.format(self._train_size))
    logger.info('Dev   question size : {}'.format(self._dev_size))
    logger.info('Test  question size : {}'.format(self._test_size))

  def _create_vocab(self, load_glove=True, maxline=-1):
    # Reconciled vocabulary
    self.word2id = {"<pad>": PAD_WID,
                    "<unk>": UNK_WID}  # word to index
    self.id2word = {PAD_WID: "<pad>",
                    UNK_WID: "<unk>"}  # index to word
    self.id2freq = {PAD_WID: -1,
                    UNK_WID: -1}  # index to word frequency in the corpus
    self.vocab_size = 2
    self.embed_matrix = None
    self.cntxt_embed_matrix = None
    self.w_embed_size = None
    # POS
    self.pos_cnt = defaultdict(int)
    self.pos2id = {"<pad>": 0}
    self.id2pos = {0: "<pad>"}
    self.num_pos_tags = 1

    logger.info("Read pretrained GloVe ...")
    gr = glove_reader.GloveReader()
    if load_glove:
      gr.read()  # NOTE: Disabled to save time debugging.

    logger.info("Count words from spacy processed dataset ...")
    word_counter = Counter()
    pos_counter = Counter()
    num_lines = sum(1 for line in open(_DATA_SOURCE))
    with codecs.open(_DATA_SOURCE, 'r', 'utf-8') as f:
      # for line_cnt, line in enumerate(f):
      for line_cnt, line in tqdm(enumerate(f), total=num_lines):
        if not line: continue
        try: prod = json.loads(line)
        except ValueError:
          excinfo = sys.exc_info()
          traceback.print_exception(excinfo[0], excinfo[1], excinfo[2])
          pdb.set_trace()

        if maxline > 0 and line_cnt > maxline: break

        for qa in prod['qa']:
          word_counter.update(qa['question'])
          word_counter.update(qa['answer'])
          pos_counter.update(qa['question_pos'])
          pos_counter.update(qa['answer_pos'])
        for review in prod['reviews']:
          word_counter.update(review['review'])

    logger.info("Reconcile with GloVe vocab ...")
    word_embeddings, cntxt_word_embeddings = [], []
    oov_tokens, rare_tokens = [], []
    idx = len(self.word2id)
    for token, token_freq in word_counter.most_common():
      if idx >= MAX_VOCAB_SIZE:
        rare_tokens.append(token)
        continue
      if token == "<pad>" or token == "<unk>":
        pdb.set_trace()
      if token in gr.vocabulary:
        word_embeddings.append(gr.embed_matrix[gr.vocabulary[token]])
        self.word2id[token] = idx
        self.id2word[idx] = token
        self.id2freq[idx] = token_freq
        idx += 1
      elif token_freq >= MIN_TOKEN_FREQ:
        oov_tokens.append(token)
        word_embeddings.append(np.random.uniform(-0.01, 0.01, gr.embed_dim))  # NOTE: NEED DOWN SCALING
        self.word2id[token] = idx
        self.id2word[idx] = token
        self.id2freq[idx] = token_freq
        idx += 1
      else:
        rare_tokens.append(token)

    assert len(self.word2id) == idx
    assert len(self.id2word) == idx
    assert len(self.id2freq) == idx
    logger.info("* OOV tokens (out of GloVe vocab): {}".format(len(oov_tokens)))
    logger.info("* Rare tokens (too rare, out of our vocab, <UNK> tokens): {}".format(len(rare_tokens)))
    logger.info("* Vocabulary size: {}".format(idx))
    self.vocab_size = idx

    idx = len(self.pos2id)
    for pos, pos_freq in pos_counter.most_common():
      self.pos2id[pos] = idx
      self.id2pos[idx] = pos
      self.num_pos_tags += 1

    # Finalize word embedding
    self.w_embed_size = gr.embed_dim
    self.embed_matrix = np.concatenate([np.zeros([2, gr.embed_dim]),
                                        preprocessing.normalize(np.stack(word_embeddings), axis=1)])
    assert (self.vocab_size, self.w_embed_size) == self.embed_matrix.shape
    del gr
    gc.collect()

  def _read_into_arrays(self, maxline=-1):
    data, review_data, asin2id = {}, {}, {}
    asin, question_id = [], []
    question_wid, question_pos, question_len = [], [], []
    answer_wid, answer_bow, answer_pos, answer_len = [], [], [], []
    review_wid, review_len, review_bow, review_avgdl, review_size_D = {}, {}, {}, {}, {}

    num_lines = sum(1 for line in open(_DATA_SOURCE))
    with codecs.open(_DATA_SOURCE, 'r', 'utf-8') as f:
      for line_cnt, line in tqdm(enumerate(f), total=num_lines):
        if not line: continue
        try: prod = json.loads(line)
        except ValueError:
          excinfo = sys.exc_info()
          traceback.print_exception(excinfo[0], excinfo[1], excinfo[2])
          pdb.set_trace()

        if maxline > 0 and line_cnt > maxline: break

        if len(prod['qa']) == 0: continue
        if len(prod['reviews']) == 0: continue

        for qa in prod['qa']:
          asin.append(prod['asin'])

          # Process questions
          question_id.append(qa['qid'])
          question_len.append(min(len(qa['question']), self.max_len))
          que_wid = self.token_to_word_id(qa['question'], exclude_unk=False)
          que_pos = [self.pos2id[_] for _ in qa['question_pos']]
          for _ in xrange(self.max_len - len(qa['question'])):  # Padding
            que_wid.append(PAD_WID)
            que_pos.append(PAD_WID)
          question_wid.append(np.asarray(que_wid[:self.max_len], dtype=np.int32))
          question_pos.append(np.asarray(que_pos[:self.max_len], dtype=np.int32))

          # Process answers
          ans_wid, ans_pos, ans_bow, ans_len = [], [], [], []
          ans_len.append(min(len(qa['answer']), self.max_len))
          wid = self.token_to_word_id(qa['answer'], exclude_unk=False)
          pos = [self.pos2id[_] for _ in qa['answer_pos']]
          for _ in xrange(self.max_len - len(qa['answer'])):  # Padding
            wid.append(PAD_WID)
            pos.append(PAD_WID)
          ans_wid.append(np.asarray(wid[:self.max_len], dtype=np.int32))
          ans_pos.append(np.asarray(pos[:self.max_len], dtype=np.int32))
          ans_bow.append(self.word_id_to_bow(self.token_to_word_id(qa['answer']), dense=False))
          answer_len.append(ans_len)
          answer_wid.append(ans_wid)
          answer_pos.append(ans_pos)
          answer_bow.append(vstack(ans_bow))

        # Process reviews
        rev_len, rev_wid, rev_pos, rev_bow = [], [], [], []
        for review in prod['reviews']:
          rev_len.append(min(len(review['review']), self.max_len))
          wid = self.token_to_word_id(review['review'], exclude_unk=False)
          for _ in xrange(self.max_len - len(review['review'])):  # Padding
            wid.append(PAD_WID)
          rev_wid.append(np.asarray(wid[:self.max_len], dtype=np.int32))
          rev_bow.append(self.word_id_to_bow(self.token_to_word_id(review['review']), dense=False))
        review_wid[prod['asin']] = np.asarray(rev_wid)
        try:
          review_bow[prod['asin']] = vstack(rev_bow)
        except ValueError:
          excinfo = sys.exc_info()
          traceback.print_exception(excinfo[0], excinfo[1], excinfo[2])
          pdb.set_trace()
        review_len[prod['asin']] = np.asarray(rev_len)
        review_avgdl[prod['asin']] = np.mean(rev_len)
        review_size_D[prod['asin']] = len(prod['reviews'])

    data['asin'] = np.stack(asin)
    del asin
    data['question_id'] = np.stack(question_id).astype(np.int32)
    del question_id
    data['question_len'] = np.stack(question_len).astype(np.int32)
    del question_len
    data['question_wid'] = np.stack(question_wid)
    del question_wid
    data['question_pos'] = np.stack(question_pos)
    del question_pos
    data['answer_len'] = np.squeeze(np.asarray(answer_len).astype(np.int32), axis=1)
    del answer_len
    data['answer_wid'] = np.squeeze(np.asarray(answer_wid), axis=1)
    del answer_wid
    data['answer_pos'] = np.squeeze(np.asarray(answer_pos), axis=1)
    del answer_pos
    data['answer_bow'] = vstack(answer_bow)
    del answer_bow
    review_data['review_wid'] = review_wid
    review_data['review_bow'] = review_bow
    review_data['review_len'] = review_len
    review_data['review_avgdl'] = review_avgdl
    review_data['review_size_D'] = review_size_D
    gc.collect()
    return data, review_data, asin2id

  def _get_review_idf(self):
    """Find the IDF score of each word in the vocabulary of all the reviewws."""
    bow = vstack([_ for _ in self.review_data['review_bow'].itervalues()])
    num_rev, vocab_size = bow.shape
    n_term = np.asarray(bow.astype(np.bool).astype(np.float32).sum(axis=0))[0]  # [vocab_size]
    idf = np.log(num_rev / np.maximum(1.0, n_term))
    assert (not np.isnan(idf).any()) and np.isfinite(idf).all()
    assert (vocab_size,) == idf.shape

    return idf

  def _data_split(self, data):
    """Random split train, dev, test sets"""
    np.random.seed(0)
    train_ratio = 0.7
    dev_ratio = 0.1

    num_ques, = data['asin'].shape
    assert data['question_id'].shape == (num_ques,)
    assert data['question_len'].shape == (num_ques,)
    assert data['question_wid'].shape == (num_ques, self.max_len)
    assert data['question_pos'].shape == (num_ques, self.max_len)
    assert data['answer_len'].shape == (num_ques,)
    assert data['answer_wid'].shape == (num_ques, self.max_len)
    assert data['answer_pos'].shape == (num_ques, self.max_len)
    assert data['answer_bow'].shape == (num_ques, self.vocab_size)  # sparse

    all_idx = np.random.permutation(num_ques)
    num_train = int(num_ques * train_ratio)
    self._train_size = num_train
    train_idx = all_idx[: num_train]
    assert len(train_idx) == self._train_size
    num_dev = int(num_ques * dev_ratio)
    self._dev_size = num_dev
    dev_idx = all_idx[num_train: num_train + num_dev]
    assert len(dev_idx) == self._dev_size
    num_test = num_ques - num_train - num_dev
    test_idx = all_idx[num_train + num_dev:]
    self._test_size = num_test
    assert len(test_idx) == self._test_size
    logger.info('Dataset split: {} train, {} dev, {} test'.format(num_train, num_dev, num_test))

    self.train_data = {'data_sample_id': np.arange(self._train_size),
                       'asin': data['asin'][train_idx],
                       'question_id': data['question_id'][train_idx],
                       'question_len': data['question_len'][train_idx],
                       'question_wid': data['question_wid'][train_idx],
                       'question_pos': data['question_pos'][train_idx],
                       'answer_len': data['answer_len'][train_idx],
                       'answer_wid': data['answer_wid'][train_idx],
                       'answer_pos': data['answer_pos'][train_idx],
                       'answer_bow': data['answer_bow'][train_idx],
                       }

    self.dev_data = {'data_sample_id': np.arange(self._dev_size),
                     'asin': data['asin'][dev_idx],
                     'question_id': data['question_id'][dev_idx],
                     'question_len': data['question_len'][dev_idx],
                     'question_wid': data['question_wid'][dev_idx],
                     'question_pos': data['question_pos'][dev_idx],
                     'answer_len': data['answer_len'][dev_idx],
                     'answer_wid': data['answer_wid'][dev_idx],
                     'answer_pos': data['answer_pos'][dev_idx],
                     'answer_bow': data['answer_bow'][dev_idx],
                     }

    self.test_data = {'data_sample_id': np.arange(self._test_size),
                      'asin': data['asin'][test_idx],
                      'question_id': data['question_id'][test_idx],
                      'question_len': data['question_len'][test_idx],
                      'question_wid': data['question_wid'][test_idx],
                      'question_pos': data['question_pos'][test_idx],
                      'answer_len': data['answer_len'][test_idx],
                      'answer_wid': data['answer_wid'][test_idx],
                      'answer_pos': data['answer_pos'][test_idx],
                      'answer_bow': data['answer_bow'][test_idx],
                      }

  def _spacy_doc_to_token(self, doc):
    que_len, tokens, POSs = 0, [], []
    delete_idx = []
    for _, w in enumerate(doc):
      t = correct_token(token(w))
      if t:
        tokens.append(t)
        POSs.append(w.pos_)
      else:
        delete_idx.append(_)
    que_len = len(tokens)
    assert que_len == len(POSs)
    return que_len, tokens, POSs, delete_idx

  def _block_to_dense(self):
    # self._train_answer_data['bow'] = self._train_answer_data['bow'].toarray().astype(np.float32)
    # self._dev_answer_data['bow'] = self._dev_answer_data['bow'].toarray().astype(np.float32)
    # self._test_answer_data['bow'] = self._test_answer_data['bow'].toarray().astype(np.float32)
    # self._answer_data['bow'] = self._answer_data['bow'].toarray().astype(np.float32)
    # self.train_data['focus_word'] = self.train_data['focus_word'].toarray().astype(np.float32)
    # self.train_data['cntxt_word'] = self.train_data['cntxt_word'].toarray().astype(np.float32)
    # self.dev_data['focus_word'] = self.dev_data['focus_word'].toarray().astype(np.float32)
    # self.dev_data['cntxt_word'] = self.dev_data['cntxt_word'].toarray().astype(np.float32)
    # self.test_data['focus_word'] = self.test_data['focus_word'].toarray().astype(np.float32)
    # self.test_data['cntxt_word'] = self.test_data['cntxt_word'].toarray().astype(np.float32)
    pass

  def _get_label(self, split, label_type, data_sample_id):
    if split == 'train':
      data_split = self.train_data
    elif split == 'dev':
      data_split = self.dev_data
    elif split == 'test':
      data_split = self.test_data
    else:
      raise ValueError('Unknown split={}'.format(split))

    if label_type == 'question_focus':
      data_split_wid = data_split['question_focus_label']
      # data_split_len = data_split['question_len']
    elif label_type == 'answer_focus':
      data_split_wid = data_split['answer_focus_label']
      # data_split_len = data_split['answer_len']
    elif label_type == 'review_focus':
      data_split_wid = data_split['review_candidate_focus_label']
      # data_split_len = data_split['review_len']

    label_seq = data_split_wid[data_sample_id]
    # len_seq = data_split_len[data_sample_id]

    # return label_seq, len_seq
    return label_seq

  def get_label(self, split):
    return lambda label_type, data_sample_id: \
      self._get_label(split, label_type, data_sample_id)

  def _get_wids(self, split, sentence_type, data_sample_id):
    if split == 'train':
      data_split = self.train_data
    elif split == 'dev':
      data_split = self.dev_data
    elif split == 'test':
      data_split = self.test_data
    else:
      raise ValueError('Unknown split={}'.format(split))

    if sentence_type == 'question':
      data_split_wid = data_split['question_wid']
      data_split_len = data_split['question_len']
    elif sentence_type == 'answer':
      data_split_wid = data_split['answer_wid']
      data_split_len = data_split['answer_len']
    elif sentence_type == 'review':
      data_split_wid = data_split['review_candidate_wid']
      data_split_len = data_split['review_candidate_len']

    wid_seq = data_split_wid[data_sample_id]
    len_seq = data_split_len[data_sample_id]

    return wid_seq, len_seq

  def get_wids(self, split):
    return lambda sentence_type, data_sample_id: \
      self._get_wids(split, sentence_type, data_sample_id)

  def _wid_seq_to_display(self, wid, exclude_pad):
    """Compatible with both 1-D and 2-D wid."""
    display = []
    if len(wid.shape) == 1:
      for i in wid:
        if not (exclude_pad and i == 0):
          display.append(self.id2word[i])
    else:
      for _wid in wid:
        display.append(self._wid_seq_to_display(_wid, exclude_pad))
    return display

  def _display_sentence(self,
                        split,
                        sentence_type,
                        data_sample_id,
                        exclude_pad=True):
    wid_seq, len_seq = self._get_wids(split=split,
                                      sentence_type=sentence_type,
                                      data_sample_id=data_sample_id)

    display = self._wid_seq_to_display(wid=wid_seq,
                                       exclude_pad=exclude_pad)

    # FIXME: assertion error when exclude_pad == False
    # try:
    #   for _, l in enumerate(len_seq):
    #     if isinstance(l, np.int32):
    #       assert len(display[_]) == l
    #     else:
    #       assert ([len(d) for d in display[_]] == l).all()
    # except ValueError:
    #   excinfo = sys.exc_info()
    #   traceback.print_exception(*excinfo)
    #   pdb.set_trace()
    return display

  def display_sentence(self, split):
    return lambda sentence_type, data_sample_id, exclude_pad=True: \
      self._display_sentence(split, sentence_type, data_sample_id, exclude_pad=exclude_pad)

  def display_review(self, asins, exclude_pad=True):
    displays = []
    for asin in asins:
      wid_seq = self.review_data['review_wid'][asin]
      len_seq = self.review_data['review_len'][asin]
      display = self._wid_seq_to_display(wid=wid_seq,
                                         exclude_pad=exclude_pad)
      displays.append(display)
    return displays

  def create_cand_review_storage(self, num_reviews=None):
    if num_reviews is None: num_reviews = self.num_reviews
    self.train_top_reviews = np.zeros([self._train_size, num_reviews], dtype=np.int32)
    self.dev_top_reviews = np.zeros([self._dev_size, num_reviews], dtype=np.int32)
    self.test_top_reviews = np.zeros([self._test_size, num_reviews], dtype=np.int32)

  def store_cand_reviews(self, split, data_sample_id, top_ids):
    assert hasattr(self, 'train_top_reviews')
    assert hasattr(self, 'dev_top_reviews')
    assert hasattr(self, 'test_top_reviews')

    if split == 'train':
      self.train_top_reviews[data_sample_id] = top_ids
    elif split == 'dev':
      self.dev_top_reviews[data_sample_id] = top_ids
    elif split == 'test':
      self.test_top_reviews[data_sample_id] = top_ids

  def save_to_disk_cand_reviews(self):
    with open(os.path.join(_DATA_PATH, 'top_reviews.pickle'), 'wb') as f:
      pickle.dump(self.train_top_reviews, f)
      pickle.dump(self.dev_top_reviews, f)
      pickle.dump(self.test_top_reviews, f)

  def load_from_disk_cand_reviews(self):
    with open(os.path.join(_DATA_PATH, 'top_reviews.pickle'), 'rb') as f:
      self.train_top_reviews = pickle.load(f)
      self.dev_top_reviews = pickle.load(f)
      self.test_top_reviews = pickle.load(f)
      assert f.read() == ''

    num_reviews = self.train_top_reviews.shape[1]
    assert self.train_top_reviews.shape == (self._train_size, num_reviews)
    assert self.dev_top_reviews.shape == (self._dev_size, num_reviews)
    assert self.test_top_reviews.shape == (self._test_size, num_reviews)

  def apply_cand_reviews(self):
    review_candidates_wid, review_candidates_pos, review_candidates_len, review_candidates_focus_label = \
      [], [], [], []
    for _ in xrange(self._train_size):
      asin = self.train_data['asin'][_]
      review_candidates_wid.append(self.review_data['review_wid'][asin][self.train_top_reviews[_]])
      review_candidates_pos.append(self.review_data['review_pos'][asin][self.train_top_reviews[_]])
      review_candidates_len.append(self.review_data['review_len'][asin][self.train_top_reviews[_]])
      review_candidates_focus_label.append(self.train_data['review_focus_label'][_][self.train_top_reviews[_]])
    self.train_data['review_candidate_wid'] = np.asarray(review_candidates_wid)
    self.train_data['review_candidate_pos'] = np.asarray(review_candidates_pos)
    self.train_data['review_candidate_len'] = np.asarray(review_candidates_len)
    self.train_data['review_candidate_focus_label'] = np.asarray(review_candidates_focus_label)
    review_candidates_wid, review_candidates_pos, review_candidates_len, review_candidates_focus_label = \
      [], [], [], []
    for _ in xrange(self._dev_size):
      asin = self.dev_data['asin'][_]
      review_candidates_wid.append(self.review_data['review_wid'][asin][self.dev_top_reviews[_]])
      review_candidates_pos.append(self.review_data['review_pos'][asin][self.dev_top_reviews[_]])
      review_candidates_len.append(self.review_data['review_len'][asin][self.dev_top_reviews[_]])
      review_candidates_focus_label.append(self.dev_data['review_focus_label'][_][self.dev_top_reviews[_]])
    self.dev_data['review_candidate_wid'] = np.asarray(review_candidates_wid)
    self.dev_data['review_candidate_pos'] = np.asarray(review_candidates_pos)
    self.dev_data['review_candidate_len'] = np.asarray(review_candidates_len)
    self.dev_data['review_candidate_focus_label'] = np.asarray(review_candidates_focus_label)
    review_candidates_wid, review_candidates_pos, review_candidates_len, review_candidates_focus_label = \
      [], [], [], []
    for _ in xrange(self._test_size):
      asin = self.test_data['asin'][_]
      review_candidates_wid.append(self.review_data['review_wid'][asin][self.test_top_reviews[_]])
      review_candidates_pos.append(self.review_data['review_pos'][asin][self.test_top_reviews[_]])
      review_candidates_len.append(self.review_data['review_len'][asin][self.test_top_reviews[_]])
      review_candidates_focus_label.append(self.test_data['review_focus_label'][_][self.test_top_reviews[_]])
    self.test_data['review_candidate_wid'] = np.asarray(review_candidates_wid)
    self.test_data['review_candidate_pos'] = np.asarray(review_candidates_pos)
    self.test_data['review_candidate_len'] = np.asarray(review_candidates_len)
    self.test_data['review_candidate_focus_label'] = np.asarray(review_candidates_focus_label)

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
          # logger.debug(u"Unknown word: {}".format(w.encode('utf-8')))
          pass
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


if __name__ == '__main__':
  logger.setLevel(logging.DEBUG)
  ch = logging.StreamHandler()
  logger.addHandler(ch)

  du = Dataset(max_len=50, selftest=True)

  pdb.set_trace()
