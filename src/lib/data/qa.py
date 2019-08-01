from __future__ import print_function
from collections import Counter
import pdb
from dataset_utils import token
# from corpus_reader import correct_token, token, bcolors
import logging
logger = logging.getLogger('root')

KEEP_ANS_MIN_WORD = 2


class bcolors:
  HEADER = u'\033[95m'
  OKBLUE = u'\033[94m'
  OKGREEN = u'\033[92m'
  WARNING = u'\033[93m'
  FAIL = u'\033[91m'
  ENDC = u'\033[0m'
  BOLD = u'\033[1m'
  UNDERLINE = u'\033[4m'


class QA:
  """A question with all it's answers."""
  def __init__(self, nlp, asin, qid, question, answers,
               if_filter_answer=True):
    self.asin = asin
    self.qid = qid
    self.question = question
    self.answers = answers
    self.q_doc = nlp(question.lower())
    
    self.a_docs = []
    num_answer = 0
    for answer in answers:
      a_doc = nlp(answer.lower())
      
      if not if_filter_answer or self._keep_answer(a_doc):
        self.a_docs.append(a_doc)
        num_answer += 1
    assert len(self.a_docs) == num_answer
    self.num_answer = num_answer
    
  def _keep_answer(self, a_doc):
    effect_w_cnt = 0
    for w in a_doc:
      if not (w.is_stop or w.is_punct or w.is_space or \
              token(w) == "yes" or token(w) == "no"):
        effect_w_cnt += 1
    if effect_w_cnt >= KEEP_ANS_MIN_WORD: 
      return True
    else:
      logger.debug(u"Filter out answer: {}".format(a_doc))
      return False

  def extract_words(self):
    self.q_token_set = set([w.lemma_.lower() for w in self.q_doc
                            if not (w.is_stop or w.is_punct or w.is_space or w.lemma_ == '-PRON-')])
    self.a_token_set = set([w.lemma_.lower() for a_doc in self.a_docs for w in a_doc
                            if not (w.is_stop or w.is_punct or w.is_space or w.lemma_ == '-PRON-')])
    
    # Question focus words are those appear in both the question and answers.
    self.question_focus_lemma = self.q_token_set & self.a_token_set

    self.q_focus_mask = [1 if (w.lemma_.lower() in self.question_focus_lemma and not w.is_stop)
                         else 0 for w in self.q_doc]
    self.a_focus_mask = [[1 if (w.lemma_.lower() in self.question_focus_lemma and not w.is_stop and not w.is_space)
                          else 0 for w in a_doc] for a_doc in self.a_docs]
    assert(len(self.a_focus_mask) == self.num_answer)

    print("{}".format(self.asin))
    print('[Question]:\t' + str(self.q_doc))
    if self.q_doc.ents:
      ents = ""
      for e in self.q_doc.ents:
        try:
          ents += "{}:{} ".format(e.text, e.label_)
        except UnicodeError:
          pass
      print(ents)
    print('[Answer]:\t\t' + str(self.a_docs))
    for a_doc in self.a_docs:
      if a_doc.ents:
        ents = ""
        for e in a_doc.ents:
          try:
            ents += "{}:{} ".format(e.text, e.label_)
          except UnicodeError:
            pass
        print(ents)
        # pdb.set_trace()
    print('[Question focus]:\t\t' + str(self.question_focus_lemma))

  def _render(self, doc, *configs):
    new_words = []
    tag = u''
    for i, w in enumerate(doc):
      flag = True
      for (mask, color) in configs:
        if mask[i]:
          new_words.append(color + w.text + bcolors.ENDC)
          flag = False
          break
      if flag: new_words.append(w.text)
    return u' '.join(new_words)

  def print_question(self):
    rendered = self._render(self.q_doc,
                            (self.q_focus_mask, bcolors.OKBLUE))
    print(rendered)

  def print_answers(self):
    for i in xrange(self.num_answer):
      rendered = self._render(self.a_docs[i],
                              (self.a_focus_mask[i], bcolors.OKBLUE),
                              (self.a_cntxt_mask[i], bcolors.OKGREEN))
      print(rendered)

  def focus_words(self):
    # Focus words may come from questions
    focus_words = [token(w) for i, w in enumerate(self.q_doc) if self.q_focus_mask[i]]
    # Focus words may also come from answers
    for aid, a_doc in enumerate(self.a_docs):
      for i, w in enumerate(a_doc):
        if self.a_focus_mask[aid][i]: focus_words.append(token(w))
    return list(set(focus_words))

  def cntxt_words(self):
    cntxt_words = []
    for aid, a_doc in enumerate(self.a_docs):
      for i, w in enumerate(a_doc):
        if self.a_cntxt_mask[aid][i]: cntxt_words.append(token(w))
    return cntxt_words

  # def count_vocab(self, vocab_cnt, pos_cnt):
  #   for w in self.q_doc:
  #     t = correct_token(token(w))
  #     if t:
  #       vocab_cnt[t] += 1
  #       pos_cnt[w.pos_] += 1
  #   for a_doc in self.a_docs:
  #     for w in a_doc:
  #       t = correct_token(token(w))
  #       if t:
  #         vocab_cnt[t] += 1
  #         pos_cnt[w.pos_] += 1
