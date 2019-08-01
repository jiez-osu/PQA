# coding: utf-8
import re
import sys
import traceback
import pdb
import numpy as np

import logging
logger = logging.getLogger('root')

LEMMATIZE = True


def token(w):
  """return a word's normalized form."""
  if LEMMATIZE: 
    if w.lemma_ == '-PRON-':
      rslt = w.text.lower()
    else:
      rslt = w.lemma_.lower()
  else: 
    rlst = w.text.lower()
  return rslt


_url_correct_pat = re.compile(ur"https?://")
_whitespace_correct_pat = re.compile(ur"\s+")
def correct_token(token):
  """Correct abnormal tokens. This is after SpaCy processing (tokenization)."""
  # Correct url
  if _url_correct_pat.search(token):
    rslt = u"<url>"
  else:
    rslt = token
  # Correct whiteshape
  rslt = _whitespace_correct_pat.sub(u'', rslt)
  if rslt:
    return rslt
  else:
    return None



_fold_pattern = re.compile(ur'»\s+Read More\s+(.*)\s+«\s+Show Less', re.UNICODE)
_lack_space_pat = re.compile(ur"(\w+)[\.!?]+([\w']+)[\s$]")
_sentence_start_lowercase_pat = re.compile(ur"(\w+)[\.?!]+\s+([a-z]\w*)[\s$]")
def correct_paragraph(text):
  """Fix problem of paragraph, which are so long that amazon generated an extra
  shortened display.
  For exmaple: Just keep the parts between "» Read More" and "« Show Less".
  NOTE: this is before SpaCy processing.
  """
  try:

    # Correct folding
    rslt = re.search(_fold_pattern, text)
    if rslt is None:
      new_text = text
    else:
      assert(len(rslt.groups()) == 1)
      new_text = rslt.groups()[0]

    # Correct unsplitted sentence
    # new_text = _lack_space_pat.sub('\g<1>. \g<2> ', new_text)
    rslt = re.search(_lack_space_pat, new_text)
    while rslt is not None:
      previous_word, latter_word = rslt.groups()
      new_words = u'{}. {} '.format(previous_word, latter_word.title())
      new_text = _lack_space_pat.sub(new_words, new_text)
      # pdb.set_trace() 
      rslt = re.search(_lack_space_pat, new_text)

    # Correct sentence without capitalization
    rslt = re.search(_sentence_start_lowercase_pat, new_text)
    while rslt is not None:
      previous_word, latter_word = rslt.groups()
      new_words = u'{}. {} '.format(previous_word, latter_word.title())
      new_text = _sentence_start_lowercase_pat.sub(new_words, new_text)
      # pdb.set_trace() 
      rslt = re.search(_sentence_start_lowercase_pat, new_text)

    return new_text
  except TypeError:
    execinfo = sys.exc_info()
    traceback.print_exception(*execinfo)
    pdb.set_trace()


def print_words(idx2word, words, word_scores=None, ignore_pad=False):
  if word_scores is not None:
    try: assert(len(words) == len(word_scores))
    except AssertionError: pdb.set_trace()
  line = u""
  for _, w in enumerate(words):
    if ignore_pad and w == 0: continue # <pad>'s word id is 0
    try:
      word = idx2word[w].encode("utf-8")
      if word_scores is None: line += u"{0} ".format(word)
      else: line += u"{0}({1:.3f}) ".format(word, word_scores[_])
    except UnicodeDecodeError:
      excinfo = sys.exc_info()
      traceback.print_exception(*excinfo)
      logger.warning("Can NOT process word: {}".format(word))
      # pdb.set_trace()
  line = line.rstrip()
  return line


def print_words_with_labels(words, labels):
  assert len(words) == len(labels)
  line = u''
  for _, (w, l) in enumerate(zip(words, labels)):
    if w == '<pad>':
      try:
        assert l == 0.
      except AssertionError:
        raise ValueError('Sentence and focus label mismatch.')
      break
    if l == 1.:
      line += u"**{0}** ".format(w)
    else:
      line += u"{0} ".format(w)
  return line


def print_words_with_scores(words, word_scores1, word_scores2=None):
  assert len(words) == len(word_scores1)
  if word_scores2 is not None:
    assert len(words) == len(word_scores2)
  line = u''
  for _, (w, s1) in enumerate(zip(words, word_scores1)):
    if w == '<pad>':
      try:
        assert s1 == 0.
        assert word_scores2 is None or word_scores2[_] == 0.
      except AssertionError:
        raise ValueError('Sentence and focus label mismatch.')
      break
    if word_scores2 is None:
      line += u"{0}({1:.2f}) ".format(w, s1)
    else:
      line += u"{0}({1:.2f}|{2:.2f}) ".format(w, s1, word_scores2[_])
  return line
    

def sample_non_answers(data_sample_id, num_sample):
  """Sample a fixed set of non-answer ids."""
  sampled_ids = []
  num_all_sample = len(data_sample_id)
  for _ in xrange(num_all_sample):
    sample_probability = np.ones(num_all_sample, dtype=np.float32)
    sample_probability[_] = 0.
    sample_probability /= (num_all_sample - 1)
    ids = np.random.choice(data_sample_id, size=num_sample, 
                           replace=False, p=sample_probability)
    sampled_ids.append(ids)
  sampled_ids = np.stack(sampled_ids)
  return sampled_ids
