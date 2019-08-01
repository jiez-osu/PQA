import tensorflow as tf

import pdb

def inter_weighted_attention(sentence,  # [batch_size, timestep, embed_size]
                             other_sentence_vec,  # [batch_size, 1, embed_size]
                             reuse=None):
  """
  Find the attention of one sequence with repect to itself and one other sentence's representation.
  Follows "Inter-Weighted Alignment Network for Sentence Pair Modeling":
  http://aclweb.org/anthology/D17-1123
  """
  bs, timestep, embed_size1 = sentence.get_shape().as_list()
  bs, ts, embed_size2 = other_sentence_vec.get_shape().as_list()
  assert ts == 1
  embed_size = embed_size1 + embed_size2

  with tf.variable_scope('inter_weighted_attention', reuse=reuse):
    inputs = tf.reshape(
        tf.concat([sentence, tf.tile(other_sentence_vec, [1, timestep, 1])], axis=2),
        [-1, embed_size])  # [batch_size * timestep, embed_size1 + embed_size2]
    hidden_size = embed_size
    w1 = tf.get_variable('weight1', [embed_size, hidden_size],
                         initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable('weight2', [hidden_size, 1],
                         initializer=tf.contrib.layers.xavier_initializer())
    hidden_output = tf.tanh(tf.matmul(inputs, w1))  # [batch_size * timestep, hidden_size]
    attention_output = tf.nn.softmax(tf.matmul(hidden_output, w2))  # [batch_size * timestep, 1]
    attention_output = tf.reshape(attention_output, [-1, timestep, 1])
    # attention = tf.squeeze(attention_output, axis=2)

  return attention_output
