from __future__ import print_function

import tensorflow as tf
import numpy as np
from lib.data.dataset import Dataset
from lib.data.dataset_utils import sample_non_answers
import cPickle as pickle

import logging
logger = logging.getLogger('root')

import pdb


def sparse_tensor(X):
  """convert sparse matrix to sparse tensor"""
  coo = X.tocoo()
  indices = np.mat([coo.row, coo.col]).transpose()
  return tf.SparseTensor(indices, coo.data, coo.shape)


def load_candidate_ids(candidate_answer_file):
  data_sample_id, inc_true_answer, answer_candidate_ids = [], [], []
  with open(candidate_answer_file, 'rb') as f:
    while True:
      try:
        data_id, inc_ans, a_ids = pickle.load(f)
        data_sample_id.append(data_id)
        inc_true_answer.append(inc_ans)
        answer_candidate_ids.append(a_ids)
      except EOFError:
        break
  data_sample_id = np.array(data_sample_id)
  inc_true_answer = np.array(inc_true_answer)
  answer_candidate_ids = np.array(answer_candidate_ids)
  return data_sample_id, inc_true_answer, answer_candidate_ids


def make_datasets_w_candidate_samples(data, batch_size, num_samples,
                                      data_sample_ids, candidate_ids,
                                      is_using_reviews=False,
                                      shuffle=False,
                                      sample='top'):
  # FIXME: This function can not support random sample training data right now.
  assert candidate_ids.shape[1] >= num_samples
  num_data_sample = len(data_sample_ids)
  assert len(data['data_sample_id']) >= num_data_sample

  if sample == 'top':
    sampled_candidate_ids = candidate_ids[:, : num_samples]
  elif sample == 'random':
    sampled_candidate_ids = np.stack([np.random.choice(ci, size=num_samples, replace=False)
                                      for ci in candidate_ids])
  with tf.device('/cpu:0'):
    tensor = {
        'data_sample_id': data['data_sample_id'][: num_data_sample],
        'question_wid': data['question_wid'][: num_data_sample],
        'question_pos': data['question_pos'][: num_data_sample].astype(np.float32),
        'question_len': data['question_len'][: num_data_sample],
        'answer_wid': data['answer_wid'][: num_data_sample],
        'answer_pos': data['answer_pos'][: num_data_sample].astype(np.float32),
        'answer_len': data['answer_len'][: num_data_sample],
        'neg_data_sample_id': data['data_sample_id'][sampled_candidate_ids],
        'neg_answer_wid': data['answer_wid'][sampled_candidate_ids],
        'neg_answer_pos': data['answer_pos'][sampled_candidate_ids].astype(np.float32),
        'neg_answer_len': data['answer_len'][sampled_candidate_ids]
    }
    assert (tensor['data_sample_id'] == data_sample_ids).all()
    if is_using_reviews:
      tensor['review_wid'] = data['review_candidate_wid'][: num_data_sample]
      tensor['review_pos'] = data['review_candidate_pos'][: num_data_sample].astype(np.float32)
      tensor['review_len'] = data['review_candidate_len'][: num_data_sample]
      tensor['review_focus_label'] = data['review_candidate_focus_label'][: num_data_sample]

    dataset = tf.data.Dataset.from_tensor_slices(tensor)
    if shuffle:
      dataset = dataset.shuffle(buffer_size=3200)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  return dataset


def make_datasets_w_random_neg_samples(data, batch_size, num_samples,
                                       is_using_reviews=False,
                                       shuffle=False):
  """This is a dataset consisting of non-answers, for training.
  Due to high extra costs, this does NOT guarantee that the true answer will not be sampled as a non-answer.

  We first follow MoQA paper "https://arxiv.org/pdf/1512.06863.pdf": where they says "In practice we use 10 epochs
  (i.e., we generate 10 random non-answers per query during each training iteration)." to generate one negative sample
  for each query at a time, and the total number of generated query are the same as the number of epochs.
  """
  neg_data_sample_id = sample_non_answers(data['data_sample_id'], num_sample=num_samples)

  with tf.device('/cpu:0'):
    tensor = {
        'data_sample_id': data['data_sample_id'],
        'question_wid': data['question_wid'],
        'question_pos': data['question_pos'].astype(np.float32),
        'question_len': data['question_len'],
        'answer_wid': data['answer_wid'],
        'answer_pos': data['answer_pos'].astype(np.float32),
        'answer_len': data['answer_len'],
        'neg_data_sample_id': data['data_sample_id'][neg_data_sample_id],
        'neg_answer_wid': data['answer_wid'][neg_data_sample_id],
        'neg_answer_pos': data['answer_pos'][neg_data_sample_id].astype(np.float32),
        'neg_answer_len': data['answer_len'][neg_data_sample_id],
    }
    if is_using_reviews:
      tensor['review_wid'] = data['review_candidate_wid']
      tensor['review_pos'] = data['review_candidate_pos'].astype(np.float32)
      tensor['review_len'] = data['review_candidate_len']
      tensor['review_focus_label'] = data['review_candidate_focus_label']

    dataset = tf.data.Dataset.from_tensor_slices(tensor)
    if shuffle:
      dataset = dataset.shuffle(buffer_size=3200)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  return dataset


def make_datasets(data, batch_size, is_using_reviews=False, shuffle=False):
  """ Define training and validation datasets with the same structure."""
  with tf.device('/cpu:0'):
    tensor = {
        'data_sample_id': data['data_sample_id'],
        'question_wid': data['question_wid'],
        'question_pos': data['question_pos'].astype(np.float32),
        'question_len': data['question_len'],
        'answer_wid': data['answer_wid'],
        'answer_pos': data['answer_pos'].astype(np.float32),
        'answer_len': data['answer_len'],
        'question_focus_label': data['question_focus_label'],
        'answer_focus_label': data['answer_focus_label'],
    }
    if is_using_reviews:
      tensor['review_wid'] = data['review_candidate_wid']
      tensor['review_pos'] = data['review_candidate_pos'].astype(np.float32)
      tensor['review_len'] = data['review_candidate_len']
      tensor['review_focus_label'] = data['review_candidate_focus_label']
    dataset = tf.data.Dataset.from_tensor_slices(tensor)
    if shuffle:
      dataset = dataset.shuffle(buffer_size=3200)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    return dataset


if __name__ == '__main__':
  data = Dataset(max_len=50, selftest=True)
  
  # CREATE DATASET
  train_dataset = make_datasets(data.train_data, 4, shuffle=True)
  dev_dataset = make_datasets(data.dev_data, 4)
  test_dataset = make_datasets(data.test_data, 4)
  handle = tf.placeholder(tf.string, shape=[])
  iterator = tf.data.Iterator.from_string_handle(
    handle, train_dataset.output_types, train_dataset.output_shapes)
  next_batch = iterator.get_next()
  train_iterator = train_dataset.make_initializable_iterator()
  dev_iterator = dev_dataset.make_initializable_iterator()
  test_iterator = test_dataset.make_initializable_iterator()

  num_epochs = 3
  with tf.Session() as sess:
    train_handle = sess.run(train_iterator.string_handle())
    dev_handle   = sess.run(dev_iterator.string_handle())
    test_handle  = sess.run(test_iterator.string_handle())

    for _ in xrange(num_epochs):
      sess.run(train_iterator.initializer)
      try:
        while True:
          value = sess.run(next_batch, feed_dict={handle: train_handle})
          # print(value)
          print("train batch")
      except tf.errors.OutOfRangeError:
        print("end of train set")

      sess.run(dev_iterator.initializer)
      try:
        while True:
          value = sess.run(next_batch, feed_dict={handle: dev_handle})
          # print(value)
          print("dev batch")
      except tf.errors.OutOfRangeError:
        print("end of dev set")

      sess.run(test_iterator.initializer)
      try:
        while True:
          value = sess.run(next_batch, feed_dict={handle: test_handle})
          # print(value)
          print("test batch")
      except tf.errors.OutOfRangeError:
        print("end of test set")







