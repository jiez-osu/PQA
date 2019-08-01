from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pdb


def question_focus_word_trgt(focus_words,     # [batch-size, vocab_size]
                             wordid_inputs):  # [batch_size, timestep]):
  batch_size, vocab_size = focus_words.get_shape().as_list()
  bs, timestep = wordid_inputs.get_shape().as_list()
  assert bs == batch_size

  # trgt = tf.stack([tf.gather(focus_words[batchid], wordid_inputs[batchid])  # [timestep]
  #                        for batchid in xrange(batch_size)])  # [batch_size, timestep]
  trgt = tf.gather_nd(
             focus_words,
             tf.stack([tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, timestep]),
                       wordid_inputs], axis=2))
  return trgt  # [batch_size, timestep]


def question_focus_word_prediction_multilayer(rnn_output,     # [batch_size, timetep, embed_size]
                                              input_lengths,  # [batch_size]
                                              reuse=None,
                                              focus_hidden_size=128):
  batch_size, timestep, embed_size = rnn_output.get_shape().as_list()
  assert [batch_size] == input_lengths.get_shape().as_list()

  with tf.variable_scope('question_focus_attention', reuse=reuse):

    # Inference
    attn_vec = tf.get_variable('weights', [embed_size, focus_hidden_size],
                               initializer=tf.contrib.layers.xavier_initializer())
    attn_bias = tf.get_variable('biases', [focus_hidden_size],
                                initializer=tf.zeros_initializer())
    
    attn_vec2 = tf.get_variable('weights2', [focus_hidden_size, 1],
                               initializer=tf.contrib.layers.xavier_initializer())
    attn_bias2 = tf.get_variable('biases2', [1],
                                initializer=tf.zeros_initializer())
    timestep_mask = tf.sequence_mask(input_lengths, maxlen=timestep, dtype=tf.float32)  # [batch_size, timestep]
    hidden = tf.tanh(tf.matmul(tf.reshape(rnn_output, [-1, embed_size]), attn_vec) + attn_bias)
    logits = timestep_mask * \
             tf.reshape(tf.matmul(hidden, attn_vec2) + attn_bias2,
                        [batch_size, timestep])  # [batch_size, timestep]
    with tf.control_dependencies([tf.check_numerics(logits, 'focus_logits')]):
      binary_pred = tf.sigmoid(logits) * timestep_mask
    
    attn = binary_pred
    attn = attn / (tf.reduce_sum(attn, axis=1, keep_dims=True) + 1e-20)  # [batch_size, timestep]
    # attn = tf.nn.softmax(logits)  # [batch_size, timestep]

    # Loss
    # losses_unmasked = tf.nn.sigmoid_cross_entropy_with_logits(labels=trgt, logits=logits)
    # losses = tf.reduce_mean(losses_unmasked * timestep_mask, axis=1)  # [batch_size]
    # losses = tf.reduce_sum(losses_unmasked * timestep_mask, axis=1) / \
    #          tf.reduce_sum(timestep_mask, axis=1)  # [batch_size]
    # loss = tf.reduce_mean(losses)
    return binary_pred, attn


def question_focus_word_prediction(rnn_output,     # [batch_size, timetep, embed_size]
                                   input_lengths,  # [batch_size]
                                   trgt,           # [batch_size, timestep]
                                   reuse=None):
  batch_size, timestep, embed_size = rnn_output.get_shape().as_list()
  assert [batch_size] == input_lengths.get_shape().as_list()

  # TODO: try ADD another layer of RNN

  with tf.variable_scope('question_focus_attention', reuse=reuse):

    # Inference
    attn_vec = tf.get_variable('weights', [embed_size, 1],
                               initializer=tf.contrib.layers.xavier_initializer())
    attn_bias = tf.get_variable('biases', [1],
                                initializer=tf.zeros_initializer())
    timestep_mask = tf.sequence_mask(input_lengths, maxlen=timestep, dtype=tf.float32)  # [batch_size, timestep]
    logits = timestep_mask * \
             tf.reshape((tf.matmul(tf.reshape(rnn_output, [-1, embed_size]), attn_vec) + attn_bias),
                        [batch_size, timestep])  # [batch_size, timestep]
    with tf.control_dependencies([tf.check_numerics(logits, 'focus_logits')]):
      binary_pred = tf.sigmoid(logits) * timestep_mask
    
    attn = binary_pred
    attn = attn / (tf.reduce_sum(attn, axis=1, keep_dims=True) + 1e-20)  # [batch_size, timestep]
    # attn = tf.nn.softmax(logits)  # [batch_size, timestep]

    # Loss
    losses_unmasked = tf.nn.sigmoid_cross_entropy_with_logits(labels=trgt, logits=logits)
    # losses = tf.reduce_mean(losses_unmasked * timestep_mask, axis=1)  # [batch_size]
    losses = tf.reduce_sum(losses_unmasked * timestep_mask, axis=1) / \
             tf.reduce_sum(timestep_mask, axis=1)  # [batch_size]
    loss = tf.reduce_mean(losses)
    return binary_pred, attn, loss


def get_top_question_focus_wordids(attn,  # [batch_size, timestep]
                                   wordid_inputs,  # [batch_size, timestep]
                                   n_focus_w=None):
  batch_size, timestep = attn.get_shape().as_list()
  assert [batch_size, timestep] == wordid_inputs.get_shape().as_list()

  if n_focus_w is None:
    n_focus_w = timestep

  with tf.control_dependencies([tf.check_numerics(attn, 'focus_attn')]):
    top_scores, top_ids = tf.nn.top_k(attn, k=n_focus_w, sorted=True)  # [batch_size, n_focus_w]
  top_words = tf.gather_nd(wordid_inputs,
                           tf.stack([tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, timestep]),
                                    top_ids], axis=2))
  return top_words, top_scores


def question_focus_prediction_metrics(binary_pred,     # [batch_size, timestep]
                                      trgt,            # [batch_size, timestep]
                                      input_lengths):  # [batch_size]
  batch_size, timestep = binary_pred.get_shape().as_list()
  assert [batch_size, timestep] == trgt.get_shape().as_list()

  # Empirical loss
  true_positive = trgt * binary_pred  # [batch_size, timestep]
  num_true_positive = tf.reduce_sum(true_positive, axis=1)  # [batch_size]
  precisions = tf.maximum(0., num_true_positive / tf.reduce_sum(binary_pred, axis=1))  # [batch_size]
  recalls = tf.maximum(0., num_true_positive / tf.reduce_sum(trgt, axis=1))  # [batch_size]
    
  #timestep_mask = tf.sequence_mask(input_lengths, maxlen=timestep, dtype=tf.float32) # [batch_size, timestep]
  # precisions, recalls = [], []
  # for batchid in xrange(batch_size):
  #   p, _ = tf.metrics.precision(labels=trgt[batchid],
  #                               predictions=binary_pred[batchid],
  #                               weights=timestep_mask[batchid]) 
  #   r, _ = tf.metrics.recall(labels=trgt[batchid],
  #                            predictions=binary_pred[batchid],
  #                            weights=timestep_mask[batchid])
  #   precisions.append(p)
  #   recalls.append(r)

  with tf.control_dependencies([tf.check_numerics(precisions, 'precision'),
                                tf.check_numerics(recalls, 'recall')]):
    prec_x_recall = precisions * recalls  # [batch_size]
    f1s = tf.where(tf.equal(prec_x_recall, tf.zeros([batch_size], dtype=tf.float32)),
                   tf.zeros([batch_size], dtype=tf.float32),
                   2. * (prec_x_recall) / (precisions + recalls)) # [batch_size]
  precision = tf.reduce_mean(precisions)
  recall = tf.reduce_mean(recalls)
  f1 = tf.reduce_mean(f1s)
  return precision, recall, f1


def review_focus_word_prediction(rnn_output,     # [batch_size, num_reviews, timestep, r_embed_size]
                                 input_lengths,  # [batch_size, num_reviews]
                                 trgt,           # [batch_size, num_reviews, timestep]
                                 reuse=None):
  batch_size, num_reviews, timestep, embed_size = rnn_output.get_shape().as_list()
  assert [batch_size, num_reviews] == input_lengths.get_shape().as_list()

  binary_pred_all, attn_all, loss_all = [], [], []
  with tf.variable_scope('review_focus_attention', reuse=reuse): # TODO: haven't modified for reviews exactly
    # Inference
    attn_vec = tf.get_variable('weights', [embed_size, 1],
                               initializer=tf.contrib.layers.xavier_initializer())
    attn_bias = tf.get_variable('biases', [1],
                                initializer=tf.zeros_initializer())
    for batchid in xrange(batch_size):
      timestep_mask = tf.sequence_mask(input_lengths[batchid], maxlen=timestep,
                                       dtype=tf.float32)  # [num_reviews, timestep]
      logits = timestep_mask * \
               tf.reshape((tf.matmul(tf.reshape(rnn_output[batchid], [-1, embed_size]), attn_vec) + attn_bias),
                          [num_reviews, timestep])  # [num_reviews, timestep]
      with tf.control_dependencies([tf.check_numerics(logits, 'review_logits')]):
          binary_pred = tf.sigmoid(logits) * timestep_mask

      attn = binary_pred
      attn = attn / (tf.reduce_sum(attn, axis=1, keep_dims=True) + 1e-20)  # [batch_size, timestep]
      # attn = tf.nn.softmax(logits)  # [batch_size, timestep]

      # Loss
      losses_unmasked = tf.nn.sigmoid_cross_entropy_with_logits(labels=trgt[batchid], logits=logits)
      # losses = tf.reduce_mean(losses_unmasked * timestep_mask, axis=1)
      losses = tf.reduce_sum(losses_unmasked * timestep_mask, axis=1) / \
               tf.reduce_sum(timestep_mask, axis=1)
      loss = tf.reduce_mean(losses)
      binary_pred_all.append(binary_pred)
      attn_all.append(attn)
      loss_all.append(loss)

    binary_pred_all = tf.stack(binary_pred_all)  # [batch_size, num_reviews, timestep]
    attn_all = tf.stack(attn_all)  # [batch_size, num_reviews, timestep]
    loss_all = tf.reduce_mean(loss)

    return binary_pred_all, attn_all, loss_all  

def get_top_review_focus_wordids(attn,           # [batch_size, num_reviews, timestep]
                                 wordid_inputs,  # [batch_size, num_reviews, timestep]
                                 n_focus_w=None):
  batch_size, num_reviews, timestep = attn.get_shape().as_list()
  assert [batch_size, num_reviews, timestep] == wordid_inputs.get_shape().as_list()

  if n_focus_w is None:
    n_focus_w = timestep

  with tf.control_dependencies([tf.check_numerics(attn, 'focus_attn')]):
    top_scores, top_ids = tf.nn.top_k(attn, k=n_focus_w, sorted=True)  # [batch_size, num_reviews, n_focus_w]
  top_words_all = []
  for batchid in xrange(batch_size):
    top_words = tf.gather_nd(wordid_inputs[batchid],
                             tf.stack([tf.tile(tf.expand_dims(tf.range(num_reviews), axis=1), [1, timestep]),
                                      top_ids[batchid]], axis=2))
  top_words_all = tf.stack(top_words_all)  # [batch_size, num_reviews, n_focus_w]
  return top_words, top_scores


def review_focus_prediction_metrics(binary_pred,     # [batch_size, num_reviews, timestep]
                                    trgt,            # [batch_size, num_reviews, timestep]
                                    input_lengths):  # [batch_size, num_reviews]
  batch_size, num_reviews, timestep = binary_pred.get_shape().as_list()
  assert [batch_size, num_reviews, timestep] == trgt.get_shape().as_list()

  # Empirical loss
  true_positive = trgt * binary_pred  # [batch_size, num_reviews, timestep]
  num_true_positive = tf.reduce_sum(true_positive, axis=2)  # [batch_size, num_reviews]
  precisions = tf.maximum(0., num_true_positive / tf.reduce_sum(binary_pred, axis=2))  # [batch_size, num_reviews]
  recalls = tf.maximum(0., num_true_positive / tf.reduce_sum(trgt, axis=2))  # [batch_size, num_reviews]]
    
  # with tf.control_dependencies([tf.check_numerics(precisions, 'precision'),
  #                               tf.check_numerics(recalls, 'recall')]):
  #   prec_x_recall = precisions * recalls  # [batch_size, num_reviews]
  #   f1s = tf.where(tf.equal(prec_x_recall, tf.zeros([batch_size, num_reviews], dtype=tf.float32)),
  #                  tf.zeros([batch_size, num_reviews], dtype=tf.float32),
  #                  2 * (prec_x_recall) / (precisions + recalls)) # [batch_size, num_reviews]
  precision = tf.reduce_mean(precisions)
  recall = tf.reduce_mean(recalls)
  # f1 = tf.reduce_mean(f1s)
  return precision, recall
  # return f1

