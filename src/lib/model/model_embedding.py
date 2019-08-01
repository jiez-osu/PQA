
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _linear(input_tensor, # [batch_size, input_size]
            input_size, output_size):
  """
  This is the word embedding method used in DSSM
  """
  weight = tf.get_variable("weights", [input_size, output_size])
  bias = tf.get_variable("biases", [1, output_size], initializer=tf.zeros_initializer())
  output = tf.tanh(tf.matmul(input_tensor, weight) + bias)
  return output


def _hash(char_hash_input, # [batch_size, input_size]
          input_size, 
          embed_size, 
          reuse=None):
  with tf.variable_scope('word_embed', reuse=reuse):
    with tf.variable_scope('hidden'):
      word_reprsnt_hidden = _linear(char_hash_input, input_size, embed_size)
    with tf.variable_scope('embed_output'):
      word_reprsnt_out = _linear(word_reprsnt_hidden, embed_size, embed_size)
  return word_reprsnt_out # [batch_size, output_size]


def trigram_hash_embed(char_hash_inputs, # timestep * [batch_size, trigram_size]
                       trigram_size,
                       w_embed_size):
  word_reprsnts = []
  with tf.variable_scope("hash_embed"):
   for i in xrange(len(char_hash_inputs)):
     word_reprsnts.append(_hash(char_hash_inputs[i],
                                input_size=trigram_size,
                                embed_size=w_embed_size,
                                reuse=True if i > 0 else None))
  return word_reprsnts
  

def lookup_embed(wordid_inputs, # [batch_size, timestep]
                 vocab_size,
                 w_embed_size,
                 initializer=None,
                 reuse=None):
  """
  This is the word embedding lookup method.
  The embedding matrix should be pretrained with word2vec algorithms.
  """
  with tf.variable_scope("lookup_embed", reuse=reuse):
    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embed_matrix", [vocab_size, w_embed_size],
          initializer=initializer, trainable=True)
      with tf.control_dependencies([tf.check_numerics(embedding, 'embedding')]):
        word_reprsnts = tf.nn.embedding_lookup(embedding, wordid_inputs)
        # FIXME: if add max_norm arg of embedding_lookup, then there is a NaN error
  return word_reprsnts # [batch_size, timestep, word_embed_size]
