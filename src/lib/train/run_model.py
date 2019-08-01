from __future__ import print_function
import numpy as np
import time
import sys
import traceback
import math
# from spacy import en
import threading
# from multiprocessing.pool import ThreadPool
import Queue
import tensorflow as tf
from itertools import compress
import cPickle as pickle

from collections import defaultdict, Counter
from lib.data.dataset_utils import print_words_with_scores, print_words_with_labels
import pdb

import logging
logger = logging.getLogger('root')


QUERY_STOP_WORDS = []
LUCENE_SPECIAL_CHARS = u"+-&|!(){}[]^\"~*?:\\"
OTHER_SPECIAL_CHARS = u"/',.@#$%"
NEG_SAMPLE_W_REVIEW = False
NEG_SAMPLE_W_NONE_ANSWER = True


data_queue = Queue.Queue(5)
rslt_queue = Queue.Queue(5)
eval_lock = threading.Lock()

def np_softmax(x, axis=0):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
  return e_x / e_x.sum(axis=axis, keepdims=True)


def f1_score(prec, reca):
  prec_x_reca = prec * reca
  if prec_x_reca == 0.0:
    f1 = 0.
  else:
    f1 = 2. * prec_x_reca / (prec + reca)
  return f1


def rerank_performance(vals, batchid, stats, at_n):
  true_answer_id = vals['data_sample_id'][batchid]
  cand_answer_id = vals['non_answer_data_sample_id'][batchid]
  pairwise_match_scores = vals['neg_pairwise_match_scores_normalized'][batchid]
  semantic_match_scores = vals['neg_semantic_match_scores_normalized'][batchid]
  ensemble_match_scores = vals['neg_ensemble_match_scores'][batchid]
  assert 'pairwise_MAP_better_cnt_at_{}'.format(at_n) in stats
  assert 'pairwise_MAP_worse_cnt_at_{}'.format(at_n) in stats
  assert 'pairwise_MAP_equal_cnt_at_{}'.format(at_n) in stats
  assert 'semantic_MAP_better_cnt_at_{}'.format(at_n) in stats
  assert 'semantic_MAP_worse_cnt_at_{}'.format(at_n) in stats
  assert 'semantic_MAP_equal_cnt_at_{}'.format(at_n) in stats
  assert 'ensemble_MAP_better_cnt_at_{}'.format(at_n) in stats
  assert 'ensemble_MAP_worse_cnt_at_{}'.format(at_n) in stats
  assert 'ensemble_MAP_equal_cnt_at_{}'.format(at_n) in stats
  assert 'MAP_na_cnt_at_{}'.format(at_n) in stats
  if len(cand_answer_id) < at_n:
    logger.warning("Candidate answer number {} is less than top-{}".format(len(cand_answer_id), at_n))

  cand_answer_id_at_n = cand_answer_id[: at_n]
  pairwise_match_scores_at_n = pairwise_match_scores[: at_n]
  semantic_match_scores_at_n = semantic_match_scores[: at_n]
  ensemble_match_scores_at_n = ensemble_match_scores[: at_n]
  error_sample_ids = set([])

  if true_answer_id in cand_answer_id_at_n:
    # CANDIDATE MAP
    true_answer_location = np.where(cand_answer_id_at_n == true_answer_id)[0][0]
    candidate_MAP = 1.0 / (1.0 + true_answer_location)
    # PAIRWISE MATCH MAP
    cnt = 0.0
    for _, s in enumerate(pairwise_match_scores_at_n):
      if pairwise_match_scores_at_n[true_answer_location] < s:
        cnt += 1.0
        error_sample_ids.add(_)
    pairwise_MAP = 1.0 / (1.0 + cnt)
    # SEMANTIC MATCH MAP
    cnt = 0.0
    for _, s in enumerate(semantic_match_scores_at_n):
      if semantic_match_scores_at_n[true_answer_location] < s:
        cnt += 1.0
        error_sample_ids.add(_)
    semantic_MAP = 1.0 / (1.0 + cnt)
    # ENSEMBLE MATCH MAP
    cnt = 0.0
    for _, s in enumerate(ensemble_match_scores_at_n):
      if ensemble_match_scores_at_n[true_answer_location] < s:
        cnt += 1.0
        error_sample_ids.add(_)
    ensemble_MAP = 1.0 / (1.0 + cnt)
    na_cnt = 0.0
  else:
    candidate_MAP = 0.0
    pairwise_MAP = 0.0
    semantic_MAP = 0.0
    ensemble_MAP = 0.0
    na_cnt = 1.0
    true_answer_location = None
    
  return na_cnt, candidate_MAP, pairwise_MAP, semantic_MAP, ensemble_MAP, \
      true_answer_location, error_sample_ids


def run_graph(session,
              model,
              handle,
              timer,
              is_training,
              print_qualitative,
              minibatch_max_num):
  cnt = 0
  while True:
    cnt += 1
    if minibatch_max_num is not None and cnt > minibatch_max_num:
      rslt_queue.put('DATA_END')
      break
    try:
      start_time = time.time()
      # Fetch results
      fetches = {
          'data_sample_id': model.data_sample_id,
          'non_answer_data_sample_id': model.non_answer_data_sample_id,
          'l2_regu_loss': model.l2_regu_loss,
          'focus_match_loss': model.focus_match_loss,
          'pairwise_match_loss': model.pairwise_match_loss,
          # 'semantic_match_loss': model.semantic_match_loss,
          'answer_focus_classify_accuracy': model.answer_focus_classify_accuracy,
          'answer_pairwise_classify_accuracy': model.answer_pairwise_classify_accuracy,
          # 'answer_semantic_classify_accuracy': model.answer_semantic_classify_accuracy,
          # 'answer_ensemble_classify_accuracy': model.answer_ensemble_classify_accuracy,
          'merged_summary_op': model.merged_summary_op,
          'global_step': model.global_step,
          'question_focus_attn': model.question_focus_attn}
      # fetches['question_focus_loss'] = model.question_focus_loss
      # fetches['question_focus_prec'] = model.question_focus_prec
      # fetches['question_focus_reca'] = model.question_focus_reca
      # fetches['question_focus_f1'] = model.question_focus_f1
      if print_qualitative:
        fetches['question_focus_binary_pred'] = model.question_focus_binary_pred
        fetches['pairwise_match_logits_2d'] = model.pairwise_match_logits_2d
        fetches['pairwise_match_logits_diff'] = model.pairwise_match_logits_diff
        # fetches['answer_attn'] = model.answer_attn
        # fetches['neg_pairwise_match_scores_normalized'] = model.neg_pairwise_match_scores_normalized
        fetches['neg_pairwise_match_scores_normalized'] = model.non_answer_pairwise_classify_logits
        # fetches['neg_semantic_match_scores_normalized'] = model.neg_semantic_match_scores_normalized
        # fetches['neg_ensemble_match_scores'] = model.neg_ensemble_match_scores
        fetches['question_exact_match'] = model.question_exact_match

      if is_training:
        fetches['eval_op'] = model.train_op
      # Feed inputs 
      feed = {model.handle: handle}
      try:
        vals = session.run(fetches=fetches, feed_dict=feed)
      except ValueError:
        excinfo = sys.exc_info()
        traceback.print_exception(*excinfo)

      timer.append(time.time() - start_time)
      # pdb.set_trace()
      rslt_queue.put(vals)
      
      # if True:  # TO DEBUG DATASET SAMPLING
      #   for id, neg_ids in zip(vals['data_sample_id'], vals['non_answer_data_sample_id']):
      #     print('{0:3d} : {1}'.format(id, ' '.join(map(str, neg_ids))))

    except tf.errors.OutOfRangeError:
      rslt_queue.put('DATA_END')
      break


def eval_stat(stats,
              display_sentence,
              get_label,
              timer,
              qsize,
              summary_writer,
              query_save_file,
              eval_rerank,
              at_n_list):
  step = 0
  while True:
    qsize.append(rslt_queue.qsize())
    rslt = rslt_queue.get()
    if rslt == "DATA_END":
      rslt_queue.put("DATA_END")
      return
    else:
      start_time = time.time()
      vals = rslt

      # SUM UP METRIC VALUES
      with eval_lock:
        stats["batch"] += 1
        stats["l2_regu_losses"] += vals["l2_regu_loss"]
        # stats["question_focus_losses"] += vals["question_focus_loss"]
        # stats["question_focus_prec"] += vals["question_focus_prec"]
        # stats["question_focus_reca"] += vals["question_focus_reca"]
        # stats["question_focus_f1"] += vals["question_focus_f1"]
        stats["focus_match_losses"] += vals["focus_match_loss"]
        stats["pairwise_match_losses"] += vals["pairwise_match_loss"]
        # stats["semantic_match_losses"] += vals["semantic_match_loss"]
        stats["answer_focus_classify_accuracy"] += vals["answer_focus_classify_accuracy"]
        stats["answer_pairwise_classify_accuracy"] += vals["answer_pairwise_classify_accuracy"]
        # stats["answer_semantic_classify_accuracy"] += vals["answer_semantic_classify_accuracy"]
        # stats["answer_ensemble_classify_accuracy"] += vals["answer_ensemble_classify_accuracy"]
        if summary_writer is not None:
          summary_writer.add_summary(vals['merged_summary_op'], vals['global_step'])
        step += 1

      # SAVE QUERY
      if query_save_file is not None:
        pickle.dump((vals['data_sample_id'], vals['question_focus_attn']), query_save_file)

      # EVALUATE SEMANTIC RERANK
      candidate_MAPs, pairwise_MAPs, semantic_MAPs, ensemble_MAPs = [], [], [], []
      true_answer_location, error_sample_ids = [], []
      if eval_rerank:
        batch_size = len(vals['data_sample_id'])
        for batchid in xrange(batch_size):
          candidate_MAPs.append([])
          pairwise_MAPs.append([])
          semantic_MAPs.append([])
          ensemble_MAPs.append([])
          for _, at_n in enumerate(at_n_list):
            na_cnt, candidate_MAP, pairwise_MAP, semantic_MAP, ensemble_MAP, \
                true_ans_loc, err_ids = \
                    rerank_performance(vals, batchid, stats, at_n)
            candidate_MAPs[-1].append(candidate_MAP)
            pairwise_MAPs[-1].append(pairwise_MAP)
            semantic_MAPs[-1].append(semantic_MAP)
            ensemble_MAPs[-1].append(ensemble_MAP)
            # SAVE RESULTS
            with eval_lock:
              stats['MAP_na_cnt_at_{}'.format(at_n)] += na_cnt
              if na_cnt == 0.0:
                if pairwise_MAP > candidate_MAP:
                  stats['pairwise_MAP_better_cnt_at_{}'.format(at_n)] += 1.0
                elif pairwise_MAP < candidate_MAP:
                  stats['pairwise_MAP_worse_cnt_at_{}'.format(at_n)] += 1.0
                else:
                  stats['pairwise_MAP_equal_cnt_at_{}'.format(at_n)] += 1.0
                if semantic_MAP > candidate_MAP:
                  stats['semantic_MAP_better_cnt_at_{}'.format(at_n)] += 1.0
                elif semantic_MAP < candidate_MAP:
                  stats['semantic_MAP_worse_cnt_at_{}'.format(at_n)] += 1.0
                else:
                  stats['semantic_MAP_equal_cnt_at_{}'.format(at_n)] += 1.0
                if ensemble_MAP > candidate_MAP:
                  stats['ensemble_MAP_better_cnt_at_{}'.format(at_n)] += 1.0
                elif ensemble_MAP < candidate_MAP:
                  stats['ensemble_MAP_worse_cnt_at_{}'.format(at_n)] += 1.0
                else:
                  stats['ensemble_MAP_equal_cnt_at_{}'.format(at_n)] += 1.0
                stats['candidate_MAP_at_{}'.format(at_n)].append(candidate_MAP)
                stats['pairwise_match_MAP_at_{}'.format(at_n)].append(pairwise_MAP)
                stats['semantic_match_MAP_at_{}'.format(at_n)].append(semantic_MAP)
                stats['ensemble_match_MAP_at_{}'.format(at_n)].append(ensemble_MAP)
              stats['candidate_MAP_at_{}_all'.format(at_n)].append(candidate_MAP)
              stats['pairwise_match_MAP_at_{}_all'.format(at_n)].append(pairwise_MAP)
              stats['semantic_match_MAP_at_{}_all'.format(at_n)].append(semantic_MAP)
              stats['ensemble_match_MAP_at_{}_all'.format(at_n)].append(ensemble_MAP)
          true_answer_location.append(true_ans_loc)
          error_sample_ids.append(err_ids)

      # DISPLAY QUALITATIVE RESULTS
      # if display_sentence is not None and get_label is not None:
      if display_sentence is not None:
        # if mode == 'question_focus':
        #   question = display_sentence('question', vals['data_sample_id'], exclude_pad=False)
        #   question_label = get_label('question_focus', vals['data_sample_id'])
        #   answer = display_sentence('answer', vals['data_sample_id'], exclude_pad=False)
        #   answer_label = get_label('answer_focus', vals['data_sample_id'])
        question = display_sentence('question', vals['data_sample_id'], exclude_pad=False)
        answer = display_sentence('answer', vals['data_sample_id'], exclude_pad=False)
        # question_label = get_label('question_focus', vals['data_sample_id'])
        # answer_label = get_label('answer_focus', vals['data_sample_id'])
        question_wo_pad = display_sentence('question', vals['data_sample_id'], exclude_pad=True)
        answer_wo_pad = display_sentence('answer', vals['data_sample_id'], exclude_pad=True)

        for batchid, (q, a) in enumerate(zip(question, answer)):
          try:
            # logger.debug('\n\n')
            # q_and_s = print_words_with_scores(q,
            #                                   vals['question_exact_match'][batchid],
            #                                   # vals['question_focus_binary_pred'][batchid],
            #                                   vals['question_focus_attn'][batchid])
            # logger.debug('[Question]:\t{}'.format(q_and_s))
            # # q_and_s = print_words_with_scores(q,
            # #                                   question_label[batchid],
            # #                                   vals['question_focus_binary_pred'][batchid],
            # #                                   vals['question_focus_attn'][batchid])
            # # logger.debug('[Question]:\t{}'.format(q_and_s))
            # # a_and_s = print_words_with_scores(a,
            # #                                   # answer_label[batchid],
            # #                                   vals['answer_attn'][batchid])
            # a_and_s = a
            # logger.debug('[Answer]:\t{}'.format(a_and_s))
            # # PRINT QUESTION QUALITATIVE
            # pw_scores = vals['pairwise_match_logits_2d'][batchid]
            # logits_diff = vals['pairwise_match_logits_diff'][batchid]  # FIXME
            # top_a_word_ids_match_q = np.flip(np.argsort(pw_scores, axis=1), axis=1)
            # # for i, (qw, l, ids) in enumerate(zip(question[batchid], 
            # #                                      question_label[batchid],
            # #                                      top_a_word_ids_match_q)):
            # for i, (qw, ids) in enumerate(zip(question[batchid], 
            #                                   top_a_word_ids_match_q)):
            #   if qw == '<pad>': break
            #   pw_match_info = '{0: <20} :\t{1: 5.4f} * {2:5.4f} = {3: 5.4f} |'.format(
            #       qw,
            #       vals['question_focus_attn'][batchid, i],
            #       logits_diff[i],
            #       vals['question_focus_attn'][batchid, i] * logits_diff[i])
            #   for _ in xrange(5):
            #     pw_match_info += '{0: <10} {1: <5.4f} | '.format(answer[batchid][ids[_]],
            #                                                      pw_scores[i, ids[_]])
            #   logger.debug(pw_match_info)
            # # # PRINT ANSWER PAIRWISE QUALITATIVE
            # # logger.debug('----- -----')
            # # top_q_word_ids_match_a = np.flip(np.argsort(pw_scores, axis=0), axis=0)
            # # for i, (aw, l, ids) in enumerate(zip(answer[batchid], 
            # #                                      answer_label[batchid], 
            # #                                      np.transpose(top_q_word_ids_match_a))):
            # #   if aw == '<pad>': break
            # #   pw_match_info = '{0: <20} :\t|'.format(aw + ('*' if l else ''))
            # #   for _ in xrange(5):
            # #     pw_match_info += '{0: <10} {1: <5.4f} | '.format(question[batchid][ids[_]],
            # #                                                      pw_scores[ids[_], i])
            # #   logger.debug(pw_match_info)

            # # # PRINT MORE ANSWER PAIRWISE QUALITATIVE
            # # logger.debug('----- -----')
            # # pw_scores_weighted = pw_scores * \
            # #                      np.expand_dims(vals['question_focus_attn'][batchid], axis=1)
            # # top_q_word_ids_match_a = np.flip(np.argsort(pw_scores_weighted, axis=0), axis=0)
            # # for i, (aw, l, ids) in enumerate(zip(answer[batchid], 
            # #                                      answer_label[batchid], 
            # #                                      np.transpose(top_q_word_ids_match_a))):
            # #   if aw == '<pad>': break
            # #   pw_match_info = '{0: <20} :\t|'.format(aw + ('*' if l else ''))
            # #   for _ in xrange(5):
            # #     pw_match_info += '{0: <10} {1: <5.4f} | '.format(question[batchid][ids[_]],
            # #                                                      pw_scores_weighted[ids[_], i])
            # #   logger.debug(pw_match_info)

            logger.debug('[Question]:\t{}'.format(q))
            logger.debug('[Answer]:\t{}'.format(a))
            pw_scores = vals['pairwise_match_logits_2d'][batchid]
            qlen = len(question_wo_pad[batchid])
            alen = len(answer_wo_pad[batchid])
            pw_scores = pw_scores[:qlen, :alen] + 10.0
            # pw_scores_normalized = pw_scores / pw_scores.sum(axis=1, keepdims=True)
            pw_scores_normalized = np_softmax(pw_scores, axis=1)

            # PRINT PAIRWISE INFORMATION (HEATMAP IF VISUALIZED)
            # pdb.set_trace()
            display_str = "\n{0:10} ".format("")
            for qw in question_wo_pad[batchid]:
              display_str += "{0:>10} ".format(qw[:10])
            display_str += "\n"
            for i, aw in enumerate(answer_wo_pad[batchid]):
              display_str += "{0:>10} ".format(aw[:10])
              for j in xrange(len(question_wo_pad[batchid])):
                display_str += "{0:10.5f} ".format(pw_scores_normalized[j, i])
              display_str += "\n"
            logger.debug(display_str)

            # PRINT RANKING QUALITATIVE
            if eval_rerank:
              non_answers = display_sentence('answer', vals['non_answer_data_sample_id'][batchid], 
                                             exclude_pad=True)
              cnt = 0
              # for _, (na, s1, s2, s3) in enumerate(zip(
              #     non_answers,
              #     vals['neg_pairwise_match_scores_normalized'][batchid],
              #     vals['neg_semantic_match_scores_normalized'][batchid],
              #     vals['neg_ensemble_match_scores'][batchid]
              #     )):
              for _, (na, s1) in enumerate(zip(non_answers, vals['neg_pairwise_match_scores_normalized'][batchid])):
                if _ > 10 and \
                   _ != true_answer_location[batchid] and \
                   _ not in error_sample_ids[batchid]:
                  continue
                # logger.debug('{0:.5f} {1:.5f} {2:.5f} | {3}'.format(s1, s2, s3, u' '.join(na)))
                logger.debug('{0:.5f} | {3}'.format(s1, u' '.join(na)))
                
              for _, at_n in enumerate(at_n_list):
                # logger.debug('MAP at {0}: '.format(at_n) +
                #              'candidate {0:.5f}; pairwise {1:.5f}; semantic {2:.5f}; ensemble {3:.5f}'.format(
                #                  candidate_MAPs[batchid][_], pairwise_MAPs[batchid][_],
                #                  semantic_MAPs[batchid][_], ensemble_MAPs[batchid][_]))
                logger.debug('MAP at {0}: '.format(at_n) +
                             'candidate {0:.5f}; pairwise {1:.5f}'.format(
                                 candidate_MAPs[batchid][_], pairwise_MAPs[batchid][_])),

          except UnicodeError:
            excinfo = sys.exc_info()
            logger.debug('Unicode error: {}'.format(excinfo[1]))
          except ValueError:
            excinfo = sys.exc_info()
            traceback.print_exception(*excinfo)
            pdb.set_trace()
          except IndexError:
            excinfo = sys.exc_info()
            traceback.print_exception(*excinfo)
            pdb.set_trace()
          except:
            traceback.print_exc()
            pdb.set_trace()

      timer.append(time.time() - start_time)


def run_epoch(session,
              model,
              data_handle,  # return a data generator
              display_sentence=None,
              get_label=None,
              is_training=False,
              query_save_file=None,
              summary_writer=None,
              minibatch_max_num=None,
              eval_rerank=False,
              at_n_list=None):
  """Runs the model on the given data."""
  start_time = time.time()
  stats = {"batch": 0.0,
           "l2_regu_losses": 0.0,
           "question_focus_losses": 0.0,
           "question_focus_prec": 0.0,
           "question_focus_reca": 0.0,
           "question_focus_f1": 0.0,
           #
           "focus_match_losses": 0.0,
           "pairwise_match_losses": 0.0,
           "semantic_match_losses": 0.0,
           "answer_focus_classify_accuracy": 0.0,
           "answer_pairwise_classify_accuracy": 0.0,
           "answer_semantic_classify_accuracy": 0.0,
           "answer_ensemble_classify_accuracy": 0.0}
  if eval_rerank:
    assert at_n_list
    for at_n in at_n_list:
      stats["candidate_MAP_at_{}".format(at_n)] = []
      stats["pairwise_match_MAP_at_{}".format(at_n)] = []
      stats["semantic_match_MAP_at_{}".format(at_n)] = []
      stats["ensemble_match_MAP_at_{}".format(at_n)] = []
      stats["candidate_MAP_at_{}_all".format(at_n)] = []
      stats["pairwise_match_MAP_at_{}_all".format(at_n)] = []
      stats["semantic_match_MAP_at_{}_all".format(at_n)] = []
      stats["ensemble_match_MAP_at_{}_all".format(at_n)] = []
      stats["pairwise_MAP_better_cnt_at_{}".format(at_n)] = 0.0
      stats["pairwise_MAP_worse_cnt_at_{}".format(at_n)] = 0.0
      stats["pairwise_MAP_equal_cnt_at_{}".format(at_n)] = 0.0
      stats["semantic_MAP_better_cnt_at_{}".format(at_n)] = 0.0
      stats["semantic_MAP_worse_cnt_at_{}".format(at_n)] = 0.0
      stats["semantic_MAP_equal_cnt_at_{}".format(at_n)] = 0.0
      stats["ensemble_MAP_better_cnt_at_{}".format(at_n)] = 0.0
      stats["ensemble_MAP_worse_cnt_at_{}".format(at_n)] = 0.0
      stats["ensemble_MAP_equal_cnt_at_{}".format(at_n)] = 0.0
      stats["MAP_na_cnt_at_{}".format(at_n)] = 0.0

  run_graph_timer, eval_stat_timers, eval_stat_qsizes = [], [], []

  # Run tensorflow graph in a separate thread
  # if display_sentence is None or get_label is None:
  if display_sentence is None:
    print_qualitative = False
  else:
    print_qualitative = True
  run_graph_thread = threading.Thread(
      target=run_graph,
      args=(session,
            model,
            data_handle,
            run_graph_timer,
            is_training,
            print_qualitative,
            minibatch_max_num
            ))
  run_graph_thread.start()

  # Evaluation in other threads
  if display_sentence is not None:  # print qualitative results
    eval_thread_num = 1
  else:
    eval_thread_num = 4
  eval_stat_threads = []
  for _ in xrange(eval_thread_num):
    eval_stat_timers.append([])
    eval_stat_qsizes.append([])
    eval_stat_threads.append(threading.Thread(
        target=eval_stat,
        args=(stats, 
              display_sentence,
              get_label,
              eval_stat_timers[_],
              eval_stat_qsizes[_],
              summary_writer,
              query_save_file,
              eval_rerank,
              at_n_list)))
  for eval_stat_thread in eval_stat_threads:
    eval_stat_thread.start()

  # Threading barrier
  run_graph_thread.join()
  for eval_stat_thread in eval_stat_threads:
    eval_stat_thread.join()
  assert (rslt_queue.qsize() == 1)
  rslt = rslt_queue.get()
  assert (rslt == "DATA_END")

  eval_stat_timer = [t for est in eval_stat_timers for t in est]
  eval_stat_qsize = [q for esq in eval_stat_qsizes for q in esq]

  logger.info("Time : %.0f seconds" % (time.time() - start_time))
  assert (len(run_graph_timer) == len(eval_stat_timer))
  logger.info("Time run graph : %f" % (np.mean(run_graph_timer)))
  logger.info("Time eval stat : %f, qsize : %.3f" % (np.mean(eval_stat_timer), np.mean(eval_stat_qsize)))
  rslt = {
      "l2_regu_loss": (stats["l2_regu_losses"] / stats["batch"]),
      "question_focus_loss": (stats["question_focus_losses"] / stats["batch"]),
      "question_focus_prec": (stats["question_focus_prec"] / stats["batch"]),
      "question_focus_reca": (stats["question_focus_reca"] / stats["batch"]),
      "question_focus_f1": (stats["question_focus_f1"] / stats["batch"]),
      #
      "focus_match_loss": (stats["focus_match_losses"] / stats["batch"]),
      "pairwise_match_loss": (stats["pairwise_match_losses"] / stats["batch"]),
      "semantic_match_loss": (stats["semantic_match_losses"] / stats["batch"]),
      "answer_focus_classify_accuracy": (stats["answer_focus_classify_accuracy"] / stats["batch"]),
      "answer_pairwise_classify_accuracy": (stats["answer_pairwise_classify_accuracy"] / stats["batch"]),
      "answer_semantic_classify_accuracy": (stats["answer_semantic_classify_accuracy"] / stats["batch"]),
      "answer_ensemble_classify_accuracy": (stats["answer_ensemble_classify_accuracy"] / stats["batch"])}

  # The following metrics evaluated how the reranking answer candidates improve the final results
  if eval_rerank:
    for at_n in at_n_list:
      rslt["candidate_MAP_at_{}".format(at_n)] = np.mean(stats["candidate_MAP_at_{}".format(at_n)])
      rslt["pairwise_match_MAP_at_{}".format(at_n)] = np.mean(stats["pairwise_match_MAP_at_{}".format(at_n)])
      rslt["semantic_match_MAP_at_{}".format(at_n)] = np.mean(stats["semantic_match_MAP_at_{}".format(at_n)])
      rslt["ensemble_match_MAP_at_{}".format(at_n)] = np.mean(stats["ensemble_match_MAP_at_{}".format(at_n)])
      rslt["candidate_MAP_at_{}_all".format(at_n)] = np.mean(stats["candidate_MAP_at_{}_all".format(at_n)])
      rslt["pairwise_match_MAP_at_{}_all".format(at_n)] = np.mean(stats["pairwise_match_MAP_at_{}_all".format(at_n)])
      rslt["semantic_match_MAP_at_{}_all".format(at_n)] = np.mean(stats["semantic_match_MAP_at_{}_all".format(at_n)])
      rslt["ensemble_match_MAP_at_{}_all".format(at_n)] = np.mean(stats["ensemble_match_MAP_at_{}_all".format(at_n)])
      rslt["pairwise_MAP_better_cnt_at_{}".format(at_n)] = stats["pairwise_MAP_better_cnt_at_{}".format(at_n)]
      rslt["pairwise_MAP_worse_cnt_at_{}".format(at_n)] = stats["pairwise_MAP_worse_cnt_at_{}".format(at_n)]
      rslt["pairwise_MAP_equal_cnt_at_{}".format(at_n)] = stats["pairwise_MAP_equal_cnt_at_{}".format(at_n)]
      rslt["semantic_MAP_better_cnt_at_{}".format(at_n)] = stats["semantic_MAP_better_cnt_at_{}".format(at_n)]
      rslt["semantic_MAP_worse_cnt_at_{}".format(at_n)] = stats["semantic_MAP_worse_cnt_at_{}".format(at_n)]
      rslt["semantic_MAP_equal_cnt_at_{}".format(at_n)] = stats["semantic_MAP_equal_cnt_at_{}".format(at_n)]
      rslt["ensemble_MAP_better_cnt_at_{}".format(at_n)] = stats["ensemble_MAP_better_cnt_at_{}".format(at_n)]
      rslt["ensemble_MAP_worse_cnt_at_{}".format(at_n)] = stats["ensemble_MAP_worse_cnt_at_{}".format(at_n)]
      rslt["ensemble_MAP_equal_cnt_at_{}".format(at_n)] = stats["ensemble_MAP_equal_cnt_at_{}".format(at_n)]
      rslt["MAP_na_cnt_at_{}".format(at_n)] = stats["MAP_na_cnt_at_{}".format(at_n)]


  return rslt

