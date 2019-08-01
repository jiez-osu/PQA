from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import sys
import traceback
import pdb

def base_layer_rnn(word_vecs,
                   input_lengths,
                   # pos_inputs=None,
                   hidden_size=None,
                   reuse=None):

  batch_size, timestep, w_embed_size = word_vecs.get_shape().as_list()
  # if pos_inputs is not None:
  #     assert [batch_size, timestep] == pos_inputs.get_shape().as_list()
  assert [batch_size] == input_lengths.get_shape().as_list()
  dynamic_batch_size = tf.shape(word_vecs)[0]
  
  with tf.variable_scope('RNN_base', reuse=reuse):
    with tf.variable_scope('GRU'):
      if hidden_size is None:
        hidden_size = w_embed_size
      fw_cell = tf.contrib.rnn.GRUCell(hidden_size)
      bw_cell = tf.contrib.rnn.GRUCell(hidden_size)
    # if pos_inputs is not None:
    #   rnn_input = tf.concat([word_vecs, tf.expand_dims(pos_inputs, axis=2)], axis=2)
    # else:
    #   rnn_input = word_vecs
    rnn_input = word_vecs
    with tf.control_dependencies([tf.check_numerics(rnn_input, 'rnn_input'),
                                  # tf.Print(rnn_input, [rnn_input], summarize=10000)
                                  ]):
      rnn_output, rnn_state = tf.nn.bidirectional_dynamic_rnn(
                                  fw_cell, bw_cell,
                                  rnn_input, sequence_length=input_lengths,
                                  initial_state_fw=fw_cell.zero_state(dynamic_batch_size, tf.float32),
                                  initial_state_bw=bw_cell.zero_state(dynamic_batch_size, tf.float32),
                                  dtype=tf.float32,
                                  swap_memory=True)
  assert [batch_size, timestep, hidden_size] == rnn_output[0].get_shape().as_list()
  assert [batch_size, timestep, hidden_size] == rnn_output[1].get_shape().as_list()
  return (rnn_output, rnn_state)


# def focus_rnn(word_reprsnts, pos_inputs, input_lengths, reuse = None):
#   word_reprsnts = tf.stack(word_reprsnts, axis=1)
#   try:
#     batch_size, timestep, w_embed_size = word_reprsnts.get_shape().as_list()
#     assert [batch_size, timestep] == pos_inputs.get_shape().as_list()
#     assert [batch_size] == input_lengths.get_shape().as_list()
#   except AssertionError:
#     excinfo = sys.exc_info()
#     traceback.print_exception(excinfo[0], excinfo[1], excinfo[2])
#     pdb.set_trace()
#
#   with tf.variable_scope('RNN_upper', reuse=reuse):
#     with tf.variable_scope('GRU'):
#       fw_cell = tf.contrib.rnn.GRUCell(w_embed_size + 1)
#       bw_cell = tf.contrib.rnn.GRUCell(w_embed_size + 1)
#     focus_rnn_input = tf.concat([word_reprsnts, tf.expand_dims(pos_inputs, axis=2)], axis=2)
#     focus_rnn_output, focus_rnn_state = tf.nn.bidirectional_dynamic_rnn(
#                                             fw_cell, bw_cell,
#                                             focus_rnn_input, sequence_length=input_lengths,
#                                             initial_state_fw=fw_cell.zero_state(batch_size, tf.float32),
#                                             initial_state_bw=bw_cell.zero_state(batch_size, tf.float32), dtype=tf.float32)
#   assert [batch_size, timestep, 2 * (w_embed_size + 1)] == focus_rnn_output[0].get_shape().as_list()
#   assert [batch_size, timestep, 2 * (w_embed_size + 1)] == focus_rnn_output[1].get_shape().as_list()
#   assert [batch_size, 2 * (w_embed_size + 1)] == focus_rnn_state.get_shape().as_list()
#   return (focus_rnn_output, focus_rnn_state)


# def upper_layer_rnn(prev_rnn_output, input_lengths, weight = None, reuse = None):
#   try:
#     batch_size, timestep, w_embed_size = prev_rnn_output[0].get_shape().as_list()
#     assert [batch_size, timestep, w_embed_size] == prev_rnn_output[1].get_shape().as_list()
#     assert [batch_size] == input_lengths.get_shape().as_list()
#     if weight is not None:
#       assert [batch_size, timestep] == weight.get_shape().as_list()
#   except AssertionError:
#     excinfo = sys.exc_info()
#     traceback.print_exception(excinfo[0], excinfo[1], excinfo[2])
#     pdb.set_trace()
#
#   with tf.variable_scope('RNN_upper', reuse=reuse):
#     if weight is None:
#       fw_rnn_input = prev_rnn_output[0]
#       bw_rnn_input = prev_rnn_output[1]
#     else:
#       scaled_weight = tf.expand_dims(tf.nn.softmax(weight, dim=-1), axis=2)
#       fw_rnn_input = scaled_weight * prev_rnn_output[0]
#       bw_rnn_input = scaled_weight * prev_rnn_output[1]
#     with tf.variable_scope('GRU'):
#       cell = tf.contrib.rnn.GRUCell(w_embed_size)
#     with tf.variable_scope('fw'):
#       fw_rnn_output, fw_rnn_state = tf.nn.dynamic_rnn(cell, fw_rnn_input, sequence_length=input_lengths, initial_state=cell.zero_state(batch_size, tf.float32), dtype=tf.float32)
#     with tf.variable_scope('bw'):
#       bw_rnn_output, bw_rnn_state = tf.nn.dynamic_rnn(cell, bw_rnn_input, sequence_length=input_lengths, initial_state=cell.zero_state(batch_size, tf.float32), dtype=tf.float32)
#   assert [batch_size, timestep, w_embed_size] == fw_rnn_output.get_shape().as_list()
#   assert [batch_size, w_embed_size] == fw_rnn_state.get_shape().as_list()
#   assert [batch_size, timestep, w_embed_size] == bw_rnn_output.get_shape().as_list()
#   assert [batch_size, w_embed_size] == bw_rnn_state.get_shape().as_list()
#   return ((fw_rnn_output, bw_rnn_output), (fw_rnn_state, bw_rnn_state))
