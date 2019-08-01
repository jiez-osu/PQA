import tensorflow as tf
import pdb


MAX_NOT_SUM = True


def mlp_match(vector_1,  # [batch_size, embed_size]
              vector_2,  # [batch_size, embed_size]
              hidden_size=None,
              reuse=None,
              scope=None,
              ):
  """Match vector 1 and vector 2 through a neural network and output the matching results."""
  batch_size, embed_size_1 = vector_1.get_shape().as_list()
  bs, embed_size_2 = vector_2.get_shape().as_list()
  assert batch_size == bs

  if hidden_size is None:
    hidden_size = embed_size_1 + embed_size_2

  match_input = tf.concat([vector_1, vector_2], axis=1)  # [batch_size, r_embed_size + q_embed_size]
  hidden_output = tf.contrib.layers.fully_connected(
    inputs=match_input,
    num_outputs=hidden_size,
    activation_fn=tf.nn.sigmoid,
    reuse=reuse,
    scope=scope + '_0')  # [batch_size, hidden_size]
  match_output = tf.contrib.layers.fully_connected(
    inputs=hidden_output,
    num_outputs=1,
    activation_fn=None,
    reuse=reuse,
    scope=scope + '_1')  # [batch_size, 1]
  return match_output


def featured_match(vector_1,  # [batch_size, embed_size]
                   vector_2,  # [batch_size, embed_size]
                   ):
  """Match vector 1 and vector 2 through a neural network and output the matching results."""
  batch_size, embed_size = vector_1.get_shape().as_list()
  bs, es = vector_2.get_shape().as_list()
  assert batch_size == bs and embed_size == es

  match_input_1 = tf.abs(vector_1 - vector_2)  # [batch_size, embed_size]
  match_input_2 = vector_1 * vector_2  # [batch_size, embed_size]

  match_input = tf.concat([match_input_1, match_input_2], axis=1)  # [batch_size, 2 * embed_size]
  match_output = tf.contrib.layers.fully_connected(
      inputs=match_input,
      num_outputs=1,
      activation_fn=tf.nn.sigmoid)  # [batch_size, 1]
  return match_output


def bilinear_match(vector1,  # [batch_size, embed_size_1]
                   vector2,  # [batch_size, embed_size_2]]
                   matrix):
  batch_size, embed_size_1 = vector1.get_shape().as_list()
  bs, embed_size_2 = vector2.get_shape().as_list()
  assert batch_size == bs

  left_combined = tf.matmul(vector1, matrix)  # [batch_size, embed_size_2]
  right_combined = tf.reduce_sum(left_combined * vector2, axis=1)  # [batch_size]
  return right_combined


def consine_match(vector_1,  # [batch_size, embed_size]
                  vector_2,  # [batch_size, embed_size]
                  ):
  batch_size, embed_size = vector_1.get_shape().as_list()
  bs, es = vector_2.get_shape().as_list()
  assert batch_size == bs and embed_size == es
  normalized_1 = tf.nn.l2_normalize(vector_1, dim=1) # [batch_size, embed_size]
  normalized_2 = tf.nn.l2_normalize(vector_2, dim=1) # [batch_size, embed_size]
  cosine_similarity = tf.reduce_sum(tf.multiply(normalized_1, normalized_2), axis=1)  # [batch_size]
  return cosine_similarity


def pairwise_match(vector1,   # [batch_size, timestep1, embed_size1]
                   vector2,   # [batch_size, timestep2, embed_size2]
                   length1,   # [batch_size]
                   length2,   # [batch_size]
                   reuse=None):
  """Match sequence of vectors1 and sequence of vectors2"""
  batch_size, timestep1, embed_size1 = vector1.get_shape().as_list()
  bs, timestep2, embed_size2 = vector2.get_shape().as_list()
  assert bs == batch_size
  assert [batch_size] == length1.get_shape().as_list()
  assert [batch_size] == length2.get_shape().as_list()


  def body(ts, ta):
    ts1 = tf.floordiv(ts, timestep2)
    ts2 = tf.mod(ts, timestep2)
    logit = consine_match(vector1[:, ts1, :],  # [batch_size, embed_size1] 
                          vector2[:, ts2, :],  # [batch_size, embed_size2]
                          )  # [batch_size]
    if MAX_NOT_SUM:
      logit = logit * 10.
    ta = ta.write(ts, logit)
    return ts + 1, ta
  
  ts = tf.constant(0)
  ta = tf.TensorArray(dtype=tf.float32, size=(timestep1 * timestep2))
  new_ts, new_ta = tf.while_loop(
      cond=lambda ts, _: ts < (timestep1 * timestep2), 
      body=body, 
      loop_vars=(ts, ta))
  
  match_logits = new_ta.stack()  # [timestep1 * timestep2, batch_size]
  match_logits_2d = tf.reshape(tf.transpose(match_logits), [batch_size, timestep1, timestep2])

  timestep_mask1 = tf.sequence_mask(length1, maxlen=timestep1, dtype=tf.float32)  # [batch_size, timestep1]
  timestep_mask2 = tf.sequence_mask(length2, maxlen=timestep2, dtype=tf.float32)  # [batch_size, timestep2]

  if MAX_NOT_SUM:
    match_logits_2d = match_logits_2d + \
                      (tf.expand_dims(timestep_mask1, axis=2) -1.) * 1e16 + \
                      (tf.expand_dims(timestep_mask2, axis=1) -1.) * 1e16
    match_logits1 = tf.reduce_max(match_logits_2d, axis=2)  # [batch_size, timestep1]
    # match_logits2 = tf.reduce_max(match_logits_2d, axis=1)  # [batch_size, timestep2]
    match_logits2 = None  # NOTE: not currently used, comment to save memory
  else:
    match_logits_2d_zero_masked = match_logits_2d * \
                                  tf.expand_dims(timestep_mask1, axis=2) * \
                                  tf.expand_dims(timestep_mask2, axis=1)
    match_logits1 = tf.reduce_sum(match_logits_2d_zero_masked, axis=2)  # [batch_size, timestep1]
    # match_logits2 = tf.reduce_sum(match_logits_2d_zero_masked, axis=1)  # [batch_size, timestep2]
    match_logits2 = None  # NOTE: not currently used, comment to save memory
    match_logits_2d = match_logits_2d + \
                      (tf.expand_dims(timestep_mask1, axis=2) -1.) * 1e16 + \
                      (tf.expand_dims(timestep_mask2, axis=1) -1.) * 1e16

  return match_logits_2d, match_logits1, match_logits2


def semantic_match(question_focus_attn,
                   q_rnn_output,
                   a_rnn_output,
                   pairwise_match_scores,
                   na_rnn_output,
                   neg_pairwise_match_scores,
                   stop_grad=False):
  batch_size, timestep, embed_size = a_rnn_output.get_shape().as_list()
  bs, num_na, ts, es = na_rnn_output.get_shape().as_list()
  assert batch_size == bs and timestep == ts and embed_size == es

  if stop_grad:
    question_focus_attn = tf.stop_gradient(question_focus_attn)
    q_rnn_output = tf.stop_gradient(q_rnn_output)
    a_rnn_output = tf.stop_gradient(a_rnn_output)
    pairwise_match_scores = tf.stop_gradient(pairwise_match_scores)

  # QUESTION AND ANSWER PAIRS SEMANTIC AGGREGATION WITH ATTENTION
  question_rep = tf.reduce_sum(q_rnn_output * tf.expand_dims(question_focus_attn, axis=2),
                               axis=1)  # [batch_size, q_embed_size]
  pairwise_attn = \
      tf.nn.softmax(pairwise_match_scores, dim=2) * \
      tf.expand_dims(question_focus_attn, axis=2)  # [batch_size, timestep, timestep]
  answer_attn = tf.reduce_sum(pairwise_attn, axis=1)  # [batch_size, timestep]
  answer_rep = tf.reduce_sum(a_rnn_output * tf.expand_dims(answer_attn, axis=2),
                               axis=1)  # [batch_size, a_embed_size]
  rep_prod_feat = question_rep * answer_rep  # [batch_size, embed_size] -- assume q_embed_size == a_embed_size
  rep_diff_feat = tf.abs(question_rep - answer_rep)  # [batch_size, embed_size] -- assume q_embed_size == a_embed_size
  # feat = tf.concat([rep_prod_feat, rep_diff_feat], axis=1)  # [batch_size, 2*embed_size]
  feat = tf.concat([rep_prod_feat, rep_diff_feat, question_rep, answer_rep], axis=1)  # [batch_size, 4*embed_size]

  # QUESTION AND NON-ANSWER PAIRS SEMANTIC AGGREGATION WITH ATTENTION
  neg_pairwise_attn = \
      tf.nn.softmax(neg_pairwise_match_scores, dim=3) * \
      tf.expand_dims(tf.expand_dims(question_focus_attn, axis=1), axis=3)  # [batch_size, num_na, timestep, timestep]
  neg_answer_attn = tf.reduce_sum(neg_pairwise_attn, axis=2)  # [batch_size, num_na, timestep]
  neg_answer_rep = tf.reduce_sum(na_rnn_output * tf.expand_dims(neg_answer_attn, axis=3),
                                 axis=2)  # [batch_size, num_na, a_embed_size]
  neg_rep_prod_feat = tf.expand_dims(question_rep, axis=1) * neg_answer_rep
  neg_rep_diff_feat = tf.abs(tf.expand_dims(question_rep, axis=1) - neg_answer_rep)
  # neg_feat = tf.concat([neg_rep_prod_feat, neg_rep_diff_feat], axis=2)  # [batch_size, num_na, 2*embed_size]
  neg_feat = tf.concat([neg_rep_prod_feat, neg_rep_diff_feat,
                        tf.tile(tf.expand_dims(question_rep, axis=1), [1, num_na, 1]),
                        neg_answer_rep], axis=2)  # [batch_size, num_na, 4*embed_size]

  all_feat = tf.concat([tf.expand_dims(feat, axis=1), neg_feat], axis=1)  # [batch_size, 1+num_na, 2*embed_size]
  answer_sem_match_hidden = tf.contrib.layers.fully_connected(
    inputs=all_feat,
    num_outputs=512,
    activation_fn=tf.tanh)  # [batch_size, 1+num_na, 128]
  answer_sem_match_hidden2 = tf.contrib.layers.fully_connected(
    inputs=answer_sem_match_hidden,
    num_outputs=256,
    activation_fn=tf.tanh)  # [batch_size, 1+num_na, 128]
  answer_sem_match_logits = tf.contrib.layers.fully_connected(
    inputs=answer_sem_match_hidden2,  # [batch_size, 1+num_na, 128]
    num_outputs=1,
    activation_fn=None)  # [batch_size, 1+num_na, 1]
  answer_sem_match_logits = tf.squeeze(answer_sem_match_logits, axis=2)  # [batch_size, 1+num_na]

  return pairwise_attn, answer_attn, answer_sem_match_logits


def semantic_match_qa_pair(question_focus_attn,     # [batch_size, timestep1]
                           q_rnn_output,            # [batch_size, timestep1, embed_size]
                           a_rnn_output,            # [batch_size, timestep2, embed_size]
                           pairwise_match_scores):  # [batch_size, timestep1, timestep2]
  batch_size, timestep1 = question_focus_attn.get_shape().as_list()
  bs, ts, embed_size = q_rnn_output.get_shape().as_list()
  assert batch_size == bs and timestep1 == ts
  bs, timestep2, es = a_rnn_output.get_shape().as_list()
  assert bs == batch_size and es == embed_size
  assert [batch_size, timestep1, timestep2] == pairwise_match_scores.get_shape().as_list()

  # PAIRWISE INTERACTION
  def body(ts, ta):
    ts1 = tf.floordiv(ts, timestep2)
    ts2 = tf.mod(ts, timestep2)
    hidden = tf.contrib.layers.fully_connected(
        tf.concat([q_rnn_output[:, ts1, :], a_rnn_output[:, ts2, :]], axis=1),  # [batch_size, 2 * embed_size]
        num_outputs=128,
        activation_fn=tf.tanh)  # [batch_size, 128]
    ta = ta.write(ts, hidden)
    return ts + 1, ta

  ts = tf.constant(0)
  ta = tf.TensorArray(dtype=tf.float32, size=(timestep1 * timestep2))
  new_ts, new_ta = tf.while_loop(
      cond=lambda ts, _ : ts < (timestep1 * timestep2),
      body=body,
      loop_vars=(ts, ta))

  interactions = new_ta.stack()  # [timestep1 * timestep2, batch_size, 128]
  interactions_2d = tf.reshape(tf.transpose(interactions, [1, 0, 2]),  # [batch_size, timestep1 * timestep2, 128]
                               [batch_size, timestep1, timestep2, 128])  # [batch_size, timestep1, timestep2, 128]
  interactions_pool_for_q = tf.reduce_max(interactions_2d, axis=2)  # [batch_size, timestep1, 128]
  interactions_pool_for_a = tf.reduce_max(interactions_2d, axis=1)  # [batch_size, timestep2, 128]

  # ANSWER ATTENTIONS
  pairwise_match_scores_normalized_for_q = tf.nn.softmax(pairwise_match_scores, dim=2)
  answer_attn = tf.reduce_sum(
      pairwise_match_scores_normalized_for_q * \
      tf.expand_dims(question_focus_attn, axis=2),  # [batch_size, timestep1, timestep2]
      axis=1)  # [batch_size, timestep2]

  # AGGREGATE SEMANTIC REPRESENTATIONS
  aggr_q_features = tf.reduce_sum(
      tf.expand_dims(question_focus_attn, axis=2) * \
      interactions_pool_for_q,  # [batch_size, timestep1, 128]
      axis=1)  # [batch_size, 128]
  aggr_a_features = tf.reduce_sum(
      tf.expand_dims(answer_attn, axis=2) * \
      interactions_pool_for_a,  # [batch_size, timestep2, 128]
      axis=1)  # [batch_size, 128]
  aggr_qa_features = tf.concat([aggr_q_features, aggr_a_features], axis=1)  # [batch_size, 256]

  # FINAL_PREDICT
  answer_sem_match_logits = tf.contrib.layers.fully_connected(
    inputs=aggr_qa_features,  # [batch_size, 256]
    num_outputs=1,
    activation_fn=None)  # [batch_size, 1]
  answer_sem_match_logits = tf.squeeze(answer_sem_match_logits, axis=1)  # [batch_size]

  return answer_attn, answer_sem_match_logits


def semantic_match_advanced(question_focus_attn,        # [batch_size, timestep1]
                            q_rnn_output,               # [batch_size, timestep1, embed_size]
                            a_rnn_output,               # [batch_size, timestep2, embed_size]
                            pairwise_match_scores,      # [batch_size, timestep1, timestep2]
                            na_rnn_output,              # [batch_size, num_na, timestep2, embed_size]
                            neg_pairwise_match_scores,  # [batch_size, num_na, timestep1, timestep2]
                            stop_grad=False):
  batch_size, timestep1 = question_focus_attn.get_shape().as_list()
  bs, ts, embed_size = q_rnn_output.get_shape().as_list()
  assert batch_size == bs and timestep1 == ts
  bs, timestep2, es = a_rnn_output.get_shape().as_list()
  assert bs == batch_size and es == embed_size
  assert [batch_size, timestep1, timestep2] == pairwise_match_scores.get_shape().as_list()
  bs, num_na, ts, es = na_rnn_output.get_shape().as_list()
  assert batch_size == bs and timestep2 == ts and embed_size == es
  assert [batch_size, num_na, timestep1, timestep2] == neg_pairwise_match_scores.get_shape().as_list()

  with tf.variable_scope('sem_match'):
    answer_attn, answer_sem_match_logits = semantic_match_qa_pair(
        question_focus_attn,  # [batch_size, timestep1]
        q_rnn_output,  # [batch_size, timestep1, embed_size]
        a_rnn_output,  # [batch_size, timestep2, embed_size]
        pairwise_match_scores)  # [batch_size, timestep1, timestep2]

  neg_answer_attn, neg_answer_sem_match_logits = [], []
  for na_id in xrange(num_na):
    with tf.variable_scope('sem_match', reuse=True):
      na_attn, na_sem_match_logits = semantic_match_qa_pair(
        question_focus_attn,      # [batch_size, timestep1]
        q_rnn_output,             # [batch_size, timestep1, embed_size]
        na_rnn_output[:, na_id],  # [batch_size, timestep2, embed_size]
        neg_pairwise_match_scores[:, na_id])  # [batch_size, timestep1, timestep2]
    neg_answer_attn.append(na_attn)
    neg_answer_sem_match_logits.append(na_sem_match_logits)

  return answer_attn, answer_sem_match_logits, neg_answer_attn, neg_answer_sem_match_logits
  



