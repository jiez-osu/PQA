from __future__ import print_function
# import sys
# import traceback
import pdb
from collections import Counter
# from corpus_reader import correct_token, token


class Reviews:
  """Reviews of a given product."""
  def __init__(self, nlp, asin, reviews, sent_split=True):
    self.asin = asin
    self.r_docs = []
    for review_para in reviews:
      try:
        # sentences = nlp(review_para.lower())
        sentences = nlp(review_para)
      except TypeError:
        pdb.set_trace()
      
      if sent_split: # Split review into sentences
        for sent in sentences.sents:
          self.r_docs.append(sent)
      else:
        self.r_docs.append(sentences)

    self.review_word_sets = self.get_word_sets(self.r_docs)

  def get_word_sets(self, r_docs):
    r_token_sets = []
    for r_doc in r_docs:
      r_token_set = set([w.lemma_.lower() for w in r_doc
                         if not (w.is_stop or w.is_punct or w.is_space or w.lemma_ == '-PRON-')])

      r_token_sets.append(r_token_set)
    assert len(r_docs) == len(r_token_sets)
    return r_token_sets

  def extract_words(self,
                    q_token_set,
                    a_token_set):
    # review word matching between questions, answers and reviews
    question_focus_lemma = q_token_set & a_token_set
    expansion_lemma_match_focus = set([])
    expansion_lemma_match_q_strong = set([])
    expansion_lemma_match_a_strong = set([])
    expansion_lemma_match_q_weak = set([])
    expansion_lemma_match_a_weak = set([])
    expansion_lemma_match_q_bridge = set([])
    expansion_lemma_match_a_bridge = set([])
    qr_match_counter = Counter([])
    r_focus_mask = []
    for _, r_token_set in enumerate(self.review_word_sets):
      qr_match = r_token_set & q_token_set
      qr_match_counter.update(qr_match)
      match_focus_set = r_token_set & question_focus_lemma
      match_question_new = (r_token_set & q_token_set) - question_focus_lemma
      match_answer_new = (r_token_set & a_token_set) - question_focus_lemma
      if_matched = False
      if match_focus_set:
        expansion_lemma_match_focus |= match_focus_set
        if match_answer_new and match_question_new:
          expansion_lemma_match_q_strong |= match_question_new
          expansion_lemma_match_a_strong |= match_answer_new
          if_matched = True
          print_msg = '[Strong]'
        elif match_answer_new or match_question_new:
          expansion_lemma_match_q_weak |= match_question_new
          expansion_lemma_match_a_weak |= match_answer_new
          if_matched = True
          print_msg = '[Weak]'
      else:
        if match_answer_new and match_question_new:
          expansion_lemma_match_q_bridge |= match_question_new
          expansion_lemma_match_a_bridge |= match_answer_new
          if_matched = True
          print_msg = '[Bridge]'

      review_focus_lemma = set([])
      if if_matched:
        review_focus_lemma |= match_question_new
        review_focus_lemma |= match_answer_new
        # print('[Review] {0:<10}'.format(print_msg) + ': ' + str(r_docs[_]))
        # print('{0:<20}'.format(' '), end=' ')
        # print(match_focus_set, end=' ')
        # print(match_question_new, end=' ')
        # print(match_answer_new)

      r_focus_mask.append(
        [1 if (w.lemma_.lower() in review_focus_lemma and not w.is_stop)
         else 0 for w in self.r_docs[_]])

    print('----')
    print('[focus_match]:\t\t', end=' ')
    print(expansion_lemma_match_focus)
    print('[Strong R-Q match]:\t\t', end=' ')
    print(expansion_lemma_match_q_strong)
    print('[Strong R-A match]:\t\t', end=' ')
    print(expansion_lemma_match_a_strong)
    print('----')
    print('[Weak R-Q match]:\t\t', end=' ')
    print(expansion_lemma_match_q_weak)
    print('[Weak R-A match]:\t\t', end=' ')
    print(expansion_lemma_match_a_weak)
    print('----')
    print('[Bridge R-Q match]:\t\t', end=' ')
    print(expansion_lemma_match_q_bridge)
    print('[Bridge R-A match]:\t\t', end=' ')
    print(expansion_lemma_match_a_bridge)
    print('----')
    print('[All Q-R match]:\t\t', end=' ')
    print(qr_match_counter.most_common(20))
    print('\n\n')

    return r_focus_mask
