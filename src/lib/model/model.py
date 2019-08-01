from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import tensorflow as tf
from lib.model.model_embedding import lookup_embed
from lib.model.model_rnn import base_layer_rnn
# from model_attention import inter_weighted_attention
# from model_match import mlp_match, featured_match
from lib.model.model_match import pairwise_match, semantic_match, semantic_match_advanced
from lib.model.model_focus import question_focus_word_trgt, question_focus_word_prediction, get_top_question_focus_wordids, \
    question_focus_prediction_metrics, review_focus_word_prediction, get_top_review_focus_wordids, \
    review_focus_prediction_metrics, question_focus_word_prediction_multilayer
import pdb

import logging

logger = logging.getLogger('root')

GATE_NONE = 0
GATE_OP = 1
GATE_GRAPH = 2

BINARY_FOCUS_ATTENTION = False
EVEN_CNTXT_ATTENTION = False
HINGE_RETRV_LOSS = True

STOP_GRAD = False
# STOP_GRAD = True

class Model(object):
  def __init__(self,
               next_batch,
               handle,
               config,
               is_training,
               mode,
               embed_initializer=None,
               randseed=None,
               ):
    self.batch_size = config.batch_size
    self.w_embed_size = config.w_embed_size
    self.vocab_size = config.vocab_size
    self.num_pos_tags = config.num_pos_tags
    self.max_len = config.max_len
    self.num_reviews = config.num_reviews  # Number of review candidates
    self.num_na = config.num_non_answers
    self.global_step = tf.get_variable('global_step', [], trainable=False,
                                       initializer=tf.constant_initializer(value=0, dtype=tf.int32))

    # Dataset inputs
    self.handle = handle
    self._data_sample_id = next_batch['data_sample_id']  # [batch_size]
    question_lengths = next_batch['question_len']  # [batch_size]
    question_wordids = next_batch['question_wid']  # [batch_size, timestep]
    question_postags = next_batch['question_pos']  # [batch_size, timestep]
    answer_lengths = next_batch['answer_len']  # [batch_size]
    answer_wordids = next_batch['answer_wid']  # [batch_size, timestep(answer)]
    answer_postags = next_batch['answer_pos']  # [batch_size, timestep(answer)]
    self.non_answer_data_sample_id = next_batch['neg_data_sample_id']  # [batch_size, num_na]
    non_answer_lengths = next_batch['neg_answer_len']  # [batch_size, num_na]
    non_answer_wordids = next_batch['neg_answer_wid']  # [batch_size, num_na, timestep(answer)]
    non_answer_postags = next_batch['neg_answer_pos']  # [batch_size, num_na, timestep(answer)]

    # EMBED WORD IDS INTO VECTORS
    q_word_vecs = lookup_embed(question_wordids, self.vocab_size, self.w_embed_size,
                               initializer=embed_initializer)  # [batch_size, timestep, w_embed_size]
    assert q_word_vecs.get_shape().as_list() == [self.batch_size, self.max_len, self.w_embed_size]
    a_word_vecs = lookup_embed(answer_wordids, self.vocab_size, self.w_embed_size,
                               initializer=embed_initializer, reuse=True)  # [batch_size, timestep, w_embed_size]
    assert a_word_vecs.get_shape().as_list() == [self.batch_size, self.max_len, self.w_embed_size]
    na_word_vecs = lookup_embed(non_answer_wordids, self.vocab_size, self.w_embed_size,
                                initializer=embed_initializer, reuse=True)  # [batch_size, num_na, timestep, w_embed_size]
    assert na_word_vecs.get_shape().as_list() == [self.batch_size, self.num_na, self.max_len, self.w_embed_size]

    # Encoding RNNs
    with tf.variable_scope('rnn'):
      q_rnn_input = tf.concat([q_word_vecs, tf.expand_dims(question_postags, axis=2)],
                              axis=2)  # [batch_size, timestep, w_embed_size + 1]
      q_wordvec_rnn_output, q_wordvec_rnn_state = base_layer_rnn(
          q_rnn_input,  # [batch_size, timestep, w_embed_size + 1]
          question_lengths,  # [batch_size]
          hidden_size=config.rnn_hidden_size,
          reuse=None)
      q_embed_size = 2 * (config.rnn_hidden_size)
      q_rnn_output = tf.concat(q_wordvec_rnn_output,  # 2 * [batch_size, timestep, w_embed_size]
                               axis=2)  # [batch_size, time_step, q_embed_size]
      q_rnn_state = tf.concat(q_wordvec_rnn_state,  # 2 * [batch_size, w_embed_size]
                              axis=1)  # [batch_size, embed_size = q_embed_size]
      assert [self.batch_size, self.max_len, q_embed_size] == q_rnn_output.get_shape().as_list()
      assert [None, q_embed_size] == q_rnn_state.get_shape().as_list()
    self.q_rnn_output = q_rnn_output
    self.q_rnn_state = q_rnn_state

    # with tf.variable_scope('answer_rnn'):
    with tf.variable_scope('rnn'):
      a_rnn_input = tf.concat([a_word_vecs, tf.expand_dims(answer_postags, axis=2)],
                              axis=2)  # [batch_size, timestep, w_embed_size + 1]
      a_wordvec_rnn_output, a_wordvec_rnn_state = base_layer_rnn(
          a_rnn_input,     # [batch_size, timestep, w_embed_size + 1]
          answer_lengths,  # [batch_size]
          hidden_size=config.rnn_hidden_size,
          reuse=True)
      a_embed_size = 2 * (config.rnn_hidden_size)
      a_rnn_output = tf.concat(a_wordvec_rnn_output,  # 2 * [2 *batch_size, timestep, w_embed_size + 1]
                               axis=2)  # [2 * batch_size, time_step, a_embed_size]
      a_rnn_state = tf.concat(a_wordvec_rnn_state,  # 2 * [2 * batch_size, w_embed_size + 1]
                              axis=1)  # [batch_size, embed_size = a_embed_size]
      assert [self.batch_size, self.max_len, a_embed_size] == a_rnn_output.get_shape().as_list()
      assert [None, a_embed_size] == a_rnn_state.get_shape().as_list()

      na_rnn_input = tf.concat([na_word_vecs, tf.expand_dims(non_answer_postags, axis=3)],
                               axis=3)  # [batch_size, num_na, timestep, w_embed_size + 1]
      na_wordvec_rnn_output_rs, na_wordvec_rnn_state_rs = base_layer_rnn(
          tf.reshape(na_rnn_input, [self.batch_size * self.num_na, self.max_len, self.w_embed_size + 1]),
          tf.reshape(non_answer_lengths, [self.batch_size * self.num_na]),
          hidden_size=config.rnn_hidden_size,
          reuse=True)
      na_rnn_output_rs = tf.concat(na_wordvec_rnn_output_rs,  # 2 * [batch_size * num_na, timestep, w_embed_size]]
                                   axis=2)  # [batch_size * num_na, timestep, a_embed_size]]
      na_rnn_state_rs = tf.concat(na_wordvec_rnn_state_rs,  # 2 * [batch_size * num_na, w_embed_size]
                                  axis=1)  # [batch_size * num_na, a_embed_size]
      na_rnn_output = tf.reshape(na_rnn_output_rs,
                                 [self.batch_size, self.num_na, self.max_len, a_embed_size])
      na_rnn_state = tf.reshape(na_rnn_state_rs,
                                [self.batch_size, self.num_na, a_embed_size])
      assert [self.batch_size, self.num_na, self.max_len, a_embed_size] == na_rnn_output.get_shape().as_list()
      assert [self.batch_size, self.num_na, a_embed_size] == na_rnn_state.get_shape().as_list()

    # PREDICT QUESTION FOCUS WORDS
    with tf.variable_scope('question_focus'):
      # question_focus_binary_pred, question_focus_attn, question_focus_loss = question_focus_word_prediction(
      #     rnn_output = q_rnn_output,  # [batch_size, time_step, q_embed_size]
      #     input_lengths = question_lengths,  # [batch_size]
      #     reuse = None)
      # question_focus_prec, question_focus_reca, question_focus_f1 = question_focus_prediction_metrics(
      #     binary_pred=question_focus_binary_pred,
      #     trgt=question_focus_labels,
      #     input_lengths=question_lengths)
      # self._question_focus_loss = question_focus_loss
      # self._question_focus_prec = question_focus_prec
      # self._question_focus_reca = question_focus_reca
      # self._question_focus_f1 = question_focus_f1
      question_focus_binary_pred, question_focus_attn = question_focus_word_prediction_multilayer(
          rnn_output=q_rnn_output,         # [batch_size, time_step, q_embed_size]
          input_lengths=question_lengths,  # [batch_size]
          reuse=None,
          focus_hidden_size=config.focus_hidden_size)
      question_top_focus_words, question_top_focus_scores = get_top_question_focus_wordids(
          attn=question_focus_attn,
          wordid_inputs=question_wordids)

      self._question_focus_binary_pred = question_focus_binary_pred
      self._question_focus_attn = question_focus_attn
      self._question_top_focus_words = question_top_focus_words
      self._question_top_focus_scores = question_top_focus_scores


      # FIND COMMON WORDS
      question_mask = tf.sequence_mask(question_lengths, maxlen=self.max_len, dtype=tf.float32)  # [batch_size, timestep]
      question_wordids_tiled = tf.tile(tf.expand_dims(question_wordids, axis=2),
                                       [1, 1, self.max_len])  # [batch_size, timestep, timestep]
      pw_exact_match = tf.equal(tf.expand_dims(answer_wordids, axis=1),
                                question_wordids_tiled)  # [batch_size, timestep, timestep]
      self.question_exact_match = tf.cast(tf.reduce_any(pw_exact_match, axis=2), tf.float32) * \
                                          question_mask  # [batch_size, timestep]
      neg_question_exact_match = []
      for _ in xrange(self.num_na):
        neg_pw_exact_match = tf.equal(tf.expand_dims(non_answer_wordids[:, _, :], axis=1),
                                      question_wordids_tiled)  # [batch_size, timestep, timestep]
        neg_question_exact_match.append(tf.cast(tf.reduce_any(neg_pw_exact_match, axis=2), tf.float32) *
                                        question_mask)  # [batch_size, timestep]
      self.neg_question_exact_match = tf.stack(neg_question_exact_match, axis=1)
      self.focus_match_score = tf.reduce_sum(self.question_exact_match *
                                             question_focus_binary_pred,
                                             axis=1)  # [batch_size]
      self.neg_focus_match_score = tf.reduce_sum(self.neg_question_exact_match *
                                                 tf.expand_dims(question_focus_binary_pred, axis=1),
                                                 axis=2)  # [batch_size, num_na]
      exp_focus_match_score = tf.exp(self.focus_match_score)  # [batch_size]
      sum_exp_neg_focus_match_score = tf.reduce_sum(tf.exp(self.neg_focus_match_score), axis=1)  #[batch_size]
      focus_match_losses = -1.0 * tf.log(exp_focus_match_score /
                                         (exp_focus_match_score + sum_exp_neg_focus_match_score))
      self.focus_match_loss = tf.reduce_mean(focus_match_losses)
      self.answer_focus_classify_accuracy = tf.reduce_mean(tf.cast(
          tf.greater(tf.expand_dims(self.focus_match_score, axis=1),
                     self.neg_focus_match_score),
          tf.float32))

    # PAIRWISE WORD SIMILARITY MATCHING FOR QA CLASSIFICATION
    with tf.variable_scope('pw_match'):
      pairwise_match_logits_2d, pairwise_match_logits, _ = \
          pairwise_match(q_rnn_output,  # [batch_size, time_step, q_embed_size]
                         a_rnn_output,  # [batch_size, time_step, a_embed_size]
                         question_lengths,  # [batch_size]
                         answer_lengths)    # [batch_size]
      weighted_pairwise_match_logits = pairwise_match_logits * question_focus_attn  # [batch_size, timestep]
      self.pairwise_match_logits_2d = pairwise_match_logits_2d

      neg_pairwise_match_logits_2d, neg_pairwise_match_logits = [], []
      weighted_neg_pairwise_match_logits = []
      for _ in xrange(self.num_na):
        neg_pw_match_logits_2d, neg_pw_match_logits, _ = \
            pairwise_match(q_rnn_output,  # [batch_size, time_step, q_embed_size]
                           na_rnn_output[:, _],  # [batch_size, time_step, a_embed_size]
                           question_lengths,   # [batch_size]
                           non_answer_lengths[:, _],  # [batch_size]
                           reuse=True)
        weighted_neg_pw_match_logits = neg_pw_match_logits * question_focus_attn
        neg_pairwise_match_logits_2d.append(neg_pw_match_logits_2d)
        neg_pairwise_match_logits.append(neg_pw_match_logits)
        weighted_neg_pairwise_match_logits.append(weighted_neg_pw_match_logits)
      neg_pairwise_match_logits_2d = \
          tf.stack(neg_pairwise_match_logits_2d, axis=1)  # [batch_size, num_na, timestep, timestep]
      neg_pairwise_match_logits = \
          tf.stack(neg_pairwise_match_logits, axis=1)  # [batch_size, num_na, timestep]
      weighted_neg_pairwise_match_logits = \
          tf.stack(weighted_neg_pairwise_match_logits, axis=1)  # [batch_size, num_na, timestep]

      self.answer_pairwise_classify_logits = \
          tf.reduce_sum(weighted_pairwise_match_logits, axis=1)  # [batch_size]
      self.non_answer_pairwise_classify_logits = \
          tf.reduce_sum(weighted_neg_pairwise_match_logits, axis=2)  # [batch_size, num_na]
      all_answer_pairwise_classify_logits = \
          tf.concat([tf.expand_dims(self.answer_pairwise_classify_logits, axis=1),
                     self.non_answer_pairwise_classify_logits], axis=1)  # [batch_size, num_na+1]

      # with tf.control_dependencies([tf.check_numerics(all_answer_pairwise_classify_logits, 'answer_pw_logits')]):
      #   all_answer_pairwise_scores_normalized = tf.nn.softmax(all_answer_pairwise_classify_logits, dim=1)

      # self.pairwise_match_scores_normalized = all_answer_pairwise_scores_normalized[:, 0]
      # self.neg_pairwise_match_scores_normalized = all_answer_pairwise_scores_normalized[:, 1:]

      # with tf.control_dependencies([tf.check_numerics(self.pairwise_match_scores_normalized, 'pw_score_normalized')]):
      #   pairwise_match_losses = -1. * tf.log(self.pairwise_match_scores_normalized)  # [batch_size]
      pairwise_match_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=tf.zeros([self.batch_size], dtype=tf.int32),
          logits=all_answer_pairwise_classify_logits)
      with tf.control_dependencies([tf.check_numerics(pairwise_match_losses, 'pw_match_losses')]):
        self.pairwise_match_loss = tf.reduce_mean(pairwise_match_losses)
      self.answer_pairwise_classify_accuracy = tf.reduce_mean(tf.cast(
          # tf.greater(tf.expand_dims(self.pairwise_match_scores_normalized, axis=1),
          #            self.neg_pairwise_match_scores_normalized),
          tf.greater(tf.expand_dims(self.answer_pairwise_classify_logits, axis=1),
                     self.non_answer_pairwise_classify_logits),
          tf.float32))
      self.pairwise_match_logits_diff = tf.reduce_max(
          tf.expand_dims(pairwise_match_logits, axis=1) - neg_pairwise_match_logits,
          axis=1)

    # FINALLY, AGGREGATE SEMANTIC INFORMATION FOR FINAL QA CLASSIFICATION
    # answer_attn, semantic_match_logits, _, neg_semantic_match_logits = \
    #     semantic_match_advanced(question_focus_attn,
    #                             q_rnn_output,
    #                             a_rnn_output,
    #                             pairwise_match_logits_2d,
    #                             na_rnn_output,
    #                             neg_pairwise_match_logits_2d,
    #                             stop_grad=STOP_GRAD)
    # self.answer_attn = answer_attn

    # all_semantic_match_scores_normalized = tf.nn.softmax(
    #     tf.stack([semantic_match_logits] + neg_semantic_match_logits, axis=1),  # [batch_size, 1 + num_na]
    #     dim=1)  # [batch_size, 1+num_na]
    # self.semantic_match_scores_normalized = all_semantic_match_scores_normalized[:, 0]
    # self.neg_semantic_match_scores_normalized = all_semantic_match_scores_normalized[:, 1:]
    # semantic_match_losses = -1.0 * tf.log(self.semantic_match_scores_normalized)
    # with tf.control_dependencies([tf.check_numerics(semantic_match_losses, 'sem_match_losses')]):
    #   self.semantic_match_loss = tf.reduce_mean(semantic_match_losses)
    # self.answer_semantic_classify_accuracy = tf.reduce_mean(tf.cast(
    #     tf.greater(tf.expand_dims(self.semantic_match_scores_normalized, axis=1),
    #                self.neg_semantic_match_scores_normalized),
    #     tf.float32))

    # SCORE ENSEMBLE RESULTS
    # self.ensemble_match_scores = self.pairwise_match_scores_normalized + \
    #                              self.semantic_match_scores_normalized
    # self.neg_ensemble_match_scores = self.neg_pairwise_match_scores_normalized + \
    #                                  self.neg_semantic_match_scores_normalized
    # ensemble_match_losses = -1 * tf.log(self.ensemble_match_scores / 2.0)
    # self.ensemble_match_loss = tf.reduce_mean(ensemble_match_losses)
    # self.answer_ensemble_classify_accuracy = tf.reduce_mean(tf.cast(
    #     tf.greater(tf.expand_dims(self.ensemble_match_scores, axis=1), self.neg_ensemble_match_scores),
    #     tf.float32))

    # SAVER
    self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    # TRAINABLE VARIABLES
    tvars = tf.trainable_variables()
    self._tvars = tvars
    logger.info('All trainable variables:')
    l2_regu_list = []
    l2_regu_list2 = []
    for v in tvars:
      if ('kernel' in v.name or 'weight' in v.name) and 'fully_connected' in v.name:
        logger.info('{0: <80}\t{1:<20}\tL2_REGU 2'.format(v.name, str(v.get_shape())))
        l2_regu_list2.append(tf.nn.l2_loss(v))
      elif ('kernel' in v.name or 'weight' in v.name) and 'fully_connected' not in v.name:
      # if ('kernel' in v.name or 'weight' in v.name):
        logger.info('{0: <80}\t{1:<20}\tL2_REGU'.format(v.name, str(v.get_shape())))
        l2_regu_list.append(tf.nn.l2_loss(v))
      else:
        logger.info('{0: <80}\t{1:<20}'.format(v.name, str(v.get_shape())))
    with tf.control_dependencies([tf.check_numerics(l2_regu_list, 'l2_regu_list')]):
      self._l2_regu_loss = tf.reduce_sum(l2_regu_list)
    with tf.control_dependencies([tf.check_numerics(l2_regu_list2, 'l2_regu_list2')]):
      self._l2_regu_loss2 = tf.reduce_sum(l2_regu_list2)

    tf.summary.scalar("l2_loss", self._l2_regu_loss)

    if not is_training:
      self._merged_summary_op = tf.summary.merge_all()
      return

    self._lr = tf.get_variable('learning_rate', [], initializer=tf.zeros_initializer(), trainable=False)
    self._new_lr = tf.placeholder(tf.float32, shape=[], name='new_learning_rate')
    self._lr_update = tf.assign(self._lr, self._new_lr)
    # self._lr_decay = tf.get_variable('learning_rate_decay', [], initializer=tf.zeros_initializer(), trainable=False)
    # self._new_lr_decay = tf.placeholder(tf.float32, shape=[], name='new_learning_rate_decay')
    # self._lr_decay_update = tf.assign(self._lr_decay, self._new_lr_decay)

    # Optimization of word matching
    loss = (self._l2_regu_loss * config.l2_regu_weight +
            self._l2_regu_loss2 * config.l2_regu_weight),
    if 'qf' in mode:
      # loss += self._question_focus_loss * 10.
      loss += self.focus_match_loss
    if 'pw' in mode:
      loss += self.pairwise_match_loss
    if 'sm' in mode:
      loss += self.semantic_match_loss
    if 'en' in mode:
      loss += self.ensemble_match_loss

    optimizer = tf.train.AdamOptimizer(self._lr)
    grads_and_vars = optimizer.compute_gradients(
        loss=loss,
        gate_gradients=GATE_OP,
        aggregation_method=None,
        colocate_gradients_with_ops=False,
        grad_loss=None)
    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    if not vars_with_grad:
      raise ValueError("No gradients provided for any variable, check your graph for ops")
    self._train_op = optimizer.apply_gradients(
      grads_and_vars,
      global_step=self.global_step, name=None)

    self._merged_summary_op = tf.summary.merge_all()


  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  # def assign_lr_decay(self, session, lr_decay_value):
  #   session.run(self._lr_decay_update, feed_dict={self._new_lr_decay: lr_decay_value})

  @property
  def train_op(self):
    return self._train_op

  @property
  def l2_regu_loss(self):
    return self._l2_regu_loss

  @property
  def l2_regu_loss2(self):
    return self._l2_regu_loss2

  @property
  def question_focus_loss(self):
    return self._question_focus_loss

  @property
  def question_focus_prec(self):
    return self._question_focus_prec

  @property
  def question_focus_reca(self):
    return self._question_focus_reca

  @property
  def question_focus_f1(self):
    return self._question_focus_f1

  @property
  def question_focus_attn(self):
    return self._question_focus_attn
 
  @property
  def question_focus_binary_pred(self):
    return self._question_focus_binary_pred
 
  @property
  def data_sample_id(self):
    return self._data_sample_id

  @property
  def saver(self):
    return self._saver

  @property
  def merged_summary_op(self):
    return self._merged_summary_op
