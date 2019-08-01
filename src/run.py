from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from datetime import datetime
import time

from lib.data.dataset import Dataset
from lib.data.tf_dataset import make_datasets, load_candidate_ids, make_datasets_w_random_neg_samples, \
    make_datasets_w_candidate_samples
from lib.model.model import Model
from lib.train.run_model import run_epoch
from lib.config.config import Config, TestConfig, SelfTestConfig, print_config
import logging
import pdb

AT_N = [10, 20, 50, 100]
AT_N_SELFTEST = [2, 5, 10]

# import pydevd
# pydevd.settrace('164.107.119.40', port=22)
# sys.path.append('/home/jzhao/Projects/amazon-qa/pycharm-debug/pycharm-debug.egg')

flags = tf.flags
flags.DEFINE_boolean("selftest", False, "If true, self test the model on tiny-scale data.")
flags.DEFINE_boolean("train", True, "If true, train; else, test.")
flags.DEFINE_boolean("test_on_train_set", True, "Run test on train split of the dataset.")
flags.DEFINE_boolean("test_on_dev_set", True, "Run test on dev split of the dataset.")
flags.DEFINE_boolean("test_on_test_set", True, "Run test on test split of the dataset.")
flags.DEFINE_string("save_path", None, "Model output directory.")
flags.DEFINE_boolean("continue_training", False, "If true, load checkpoints rather than randomly initialize params.")
flags.DEFINE_integer("random_seed", 0, "Random seed.")
flags.DEFINE_integer("tf_random_seed", 0, "Random seed.")
flags.DEFINE_string("checkpoint_path", "./../checkpoints", "Model parameter checkpoint directory.")
flags.DEFINE_string("log_file_name", "exp", "Name of the log file")
flags.DEFINE_boolean("write_summary", False, "If true, write summary.")
flags.DEFINE_boolean("save_query", False, "If true, save query into file.")
flags.DEFINE_string("gpu_id", "0", "Which GPU to run the model on.")
flags.DEFINE_boolean("allow_gpu_memory_growth", False, "Allow dynamic GPU memory usage.")
flags.DEFINE_boolean("use_candidate_answers", False, "If true, use candidate answers from search")
flags.DEFINE_string("load_checkpoint", "pairwise", "[focus | pairwise | semantic | ensemble]")
flags.DEFINE_boolean("add_focus_loss", True, "Include question focus loss in training")
flags.DEFINE_boolean("add_pairwise_loss", True, "Include pairwise match loss in training")
flags.DEFINE_boolean("add_semantic_loss", False, "Include semantic match loss in training")
flags.DEFINE_boolean("add_ensemble_loss", False, "Include ensemble match loss in training")
flags.DEFINE_boolean("print_test_qualitative", True, "Print qualitative results to log file during test.")
flags.DEFINE_boolean("resample_train_negative", True, "If True, re-sample negative samples for training.")
#
flags.DEFINE_integer("config_num_non_answers", None, "If positive, overwrite config.")
flags.DEFINE_integer("config_batch_size", None, "If positive, overwrite config.")
flags.DEFINE_float("config_learning_rate", None, "Overwrite Config when not none.")

FLAGS = flags.FLAGS

datetime_string = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file_name = ('selftest_' if FLAGS.selftest else '') + \
                ('train_' if FLAGS.train else 'test_') + \
                FLAGS.log_file_name + '_' + datetime_string + '.log'
log_file_dir = './../logs'
if not os.path.exists(log_file_dir): os.makedirs(log_file_dir)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(module)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=os.path.join(log_file_dir, log_file_name),
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(module)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logger = logging.getLogger('root')
logger.addHandler(console)


def print_flags():
  msg = ""
  for flg, value in sorted(tf.flags.FLAGS.__flags.iteritems(), key=lambda x: x[0]):
    msg += "\n\t\t{0:<40s} : {1}".format(flg, value)
  logger.info(msg)


def get_config(data):
  if FLAGS.selftest:
    config = SelfTestConfig()
  elif FLAGS.train:
    config = Config()
  else:
    config = TestConfig()
  config.vocab_size = data.vocab_size
  config.num_pos_tags = data.num_pos_tags
  config.w_embed_size = data.w_embed_size
  config.max_len = data.max_len
  if FLAGS.config_learning_rate is not None:
    config.learning_rate = FLAGS.config_learning_rate
  if FLAGS.config_num_non_answers is not None:
    config.num_non_answers = FLAGS.config_num_non_answers
  if FLAGS.config_batch_size is not None:
    config.batch_size = FLAGS.config_batch_size
  return config


def load_ckpt(sess, model, ckptpath):
  ckpt = tf.train.get_checkpoint_state(ckptpath)
  assert(ckpt)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    logger.info("Loading checkpoint from {} ...".format(ckptpath))
    model.saver.restore(sess, ckpt.model_checkpoint_path)


def save_ckpt(sess, model, ckptpath):
  path = os.path.join(ckptpath, 'model.ckpt')
  if not os.path.exists(os.path.dirname(ckptpath)):
    try:
      os.makedirs(os.path.dirname(ckptpath))
    except OSError as exc:  # Guard against race condition
      if exc.errno != errno.EEXIST:
        raise
  global_step = sess.run(model.global_step)
  logger.info(">>>>> >>>>> Saving a checkpoint to {0} (global_step={1:d}) >>>>> >>>>>".format(
      ckptpath, int(global_step)))
  model.saver.save(sess, path, global_step=model.global_step,
                   write_meta_graph=False)

# def show_rerank_results(result):
#   # PAIRWISE MATCH RE-RANK RESULTS
#   cnt_in = result['pairwise_MAP_better_cnt'] + \
#            result['pairwise_MAP_worse_cnt'] + \
#            result['pairwise_MAP_equal_cnt']
#   assert cnt_in == result['semantic_MAP_better_cnt'] + \
#          result['semantic_MAP_worse_cnt'] + \
#          result['semantic_MAP_equal_cnt']
#   assert cnt_in == result['ensemble_MAP_better_cnt'] + \
#          result['ensemble_MAP_worse_cnt'] + \
#          result['ensemble_MAP_equal_cnt']
#   cnt_all = cnt_in + result['MAP_na_cnt']
#   logger.info("True answer in candidate set: {0} / {1} ({2:.5f})".format(cnt_in, cnt_all, cnt_in / cnt_all))
#   if not (cnt_in == 0.0 or cnt_all == 0.0):
#     logger.info("MAP before / after PAIRWISE rerank {0:7.5f} ({1:7.5f}) / {2:7.5f} ({3:7.5f})".format(
#       result['candidate_MAP'], result['pairwise_match_MAP'],
#       result['candidate_MAP_all'], result['pairwise_match_MAP_all']))
#     logger.info("True answer ranked higher / lower / same:\t{0} / {1} / {2}".format(
#       result['pairwise_MAP_better_cnt'],
#       result['pairwise_MAP_worse_cnt'],
#       result['pairwise_MAP_equal_cnt']))
#   # SEMANTIC MATCH RE-RANK RESULTS
#   if not (cnt_in == 0.0 or cnt_all == 0.0):
#     logger.info("MAP before / after SEMANTIC rerank {0:7.5f} ({1:7.5f}) / {2:7.5f} ({3:7.5f})".format(
#       result['candidate_MAP'], result['semantic_match_MAP'],
#       result['candidate_MAP_all'], result['semantic_match_MAP_all']))
#     logger.info("True answer ranked higher / lower / same:\t{0} / {1} / {2}".format(
#       result['semantic_MAP_better_cnt'],
#       result['semantic_MAP_worse_cnt'],
#       result['semantic_MAP_equal_cnt']))
#   # ENSEMBLE MATCH RE-RANK RESULTS
#   if not (cnt_in == 0.0 or cnt_all == 0.0):
#     logger.info("MAP before / after ENSEMBLE rerank {0:7.5f} ({1:7.5f}) / {2:7.5f} ({3:7.5f})".format(
#       result['candidate_MAP'], result['ensemble_match_MAP'],
#       result['candidate_MAP_all'], result['ensemble_match_MAP_all']))
#     logger.info("True answer ranked higher / lower / same:\t{0} / {1} / {2}".format(
#       result['ensemble_MAP_better_cnt'],
#       result['ensemble_MAP_worse_cnt'],
#       result['ensemble_MAP_equal_cnt']))


def show_rerank_results(result):
  for at_n in (AT_N_SELFTEST if FLAGS.selftest else AT_N):
    logger.info("Rerank top {}:".format(at_n))
    # PAIRWISE MATCH RE-RANK RESULTS
    cnt_in = result['pairwise_MAP_better_cnt_at_{}'.format(at_n)] + \
             result['pairwise_MAP_worse_cnt_at_{}'.format(at_n)] + \
             result['pairwise_MAP_equal_cnt_at_{}'.format(at_n)]
    assert cnt_in == result['semantic_MAP_better_cnt_at_{}'.format(at_n)] + \
           result['semantic_MAP_worse_cnt_at_{}'.format(at_n)] + \
           result['semantic_MAP_equal_cnt_at_{}'.format(at_n)]
    assert cnt_in == result['ensemble_MAP_better_cnt_at_{}'.format(at_n)] + \
           result['ensemble_MAP_worse_cnt_at_{}'.format(at_n)] + \
           result['ensemble_MAP_equal_cnt_at_{}'.format(at_n)]
    cnt_all = cnt_in + result['MAP_na_cnt_at_{}'.format(at_n)]
    logger.info("True answer in candidate set: {0} / {1} ({2:.5f})".format(cnt_in, cnt_all, cnt_in / cnt_all))
    if not (cnt_in == 0.0 or cnt_all == 0.0):
      logger.info("MAP before / after PAIRWISE rerank {0:7.5f} / {1:7.5f} ; {2:7.5f} / {3:7.5f}".format(
        result['candidate_MAP_at_{}'.format(at_n)], result['pairwise_match_MAP_at_{}'.format(at_n)],
        result['candidate_MAP_at_{}_all'.format(at_n)], result['pairwise_match_MAP_at_{}_all'.format(at_n)]))
      logger.info("True answer ranked higher / lower / same:\t{0} / {1} / {2}".format(
        result['pairwise_MAP_better_cnt_at_{}'.format(at_n)],
        result['pairwise_MAP_worse_cnt_at_{}'.format(at_n)],
        result['pairwise_MAP_equal_cnt_at_{}'.format(at_n)]))
    # SEMANTIC MATCH RE-RANK RESULTS
    if not (cnt_in == 0.0 or cnt_all == 0.0):
      logger.info("MAP before / after SEMANTIC rerank {0:7.5f} / {1:7.5f} ; {2:7.5f} / {3:7.5f}".format(
        result['candidate_MAP_at_{}'.format(at_n)], result['semantic_match_MAP_at_{}'.format(at_n)],
        result['candidate_MAP_at_{}_all'.format(at_n)], result['semantic_match_MAP_at_{}_all'.format(at_n)]))
      logger.info("True answer ranked higher / lower / same:\t{0} / {1} / {2}".format(
        result['semantic_MAP_better_cnt_at_{}'.format(at_n)],
        result['semantic_MAP_worse_cnt_at_{}'.format(at_n)],
        result['semantic_MAP_equal_cnt_at_{}'.format(at_n)]))
    # ENSEMBLE MATCH RE-RANK RESULTS
    if not (cnt_in == 0.0 or cnt_all == 0.0):
      logger.info("MAP before / after ENSEMBLE rerank {0:7.5f} / {1:7.5f} ; {2:7.5f} / {3:7.5f}".format(
        result['candidate_MAP_at_{}'.format(at_n)], result['ensemble_match_MAP_at_{}'.format(at_n)],
        result['candidate_MAP_at_{}_all'.format(at_n)], result['ensemble_match_MAP_at_{}_all'.format(at_n)]))
      logger.info("True answer ranked higher / lower / same:\t{0} / {1} / {2}".format(
        result['ensemble_MAP_better_cnt_at_{}'.format(at_n)],
        result['ensemble_MAP_worse_cnt_at_{}'.format(at_n)],
        result['ensemble_MAP_equal_cnt_at_{}'.format(at_n)]))


def test():
  np.random.seed(FLAGS.random_seed)
  data = Dataset(max_len=50,
                 num_reviews=0,
                 selftest=FLAGS.selftest,
                 load_review=False)
  test_config = get_config(data)
  logger.info("Test config:" + print_config(test_config))
  mode = []
  # if FLAGS.add_focus_loss:
  #   mode.append('qf')
  # if FLAGS.add_pairwise_loss:
  #   mode.append('pw')
  # if FLAGS.add_semantic_loss:
  #   mode.append('sm')
  # if FLAGS.add_ensemble_loss:
  #   mode.append('en')

  # CONFIGURE PATHS
  query_path = os.path.join('./../intermediate', '{}-query{}'.format(
      FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
  if not os.path.exists(query_path):
    os.makedirs(query_path)
  if not FLAGS.continue_training:
    checkpoint_path_qf_load = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-qf{}'.format(
        FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
    checkpoint_path_pw_load = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-pw{}'.format(
        FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
    checkpoint_path_sm_load = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-sm{}'.format(
        FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
    checkpoint_path_en_load = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-en{}'.format(
        FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
  else:
    checkpoint_path_qf_load = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-qf{}-cont'.format(
      FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
    checkpoint_path_pw_load = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-pw{}-cont'.format(
      FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
    checkpoint_path_sm_load = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-sm{}-cont'.format(
      FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
    checkpoint_path_en_load = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-en{}-cont'.format(
      FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))

  with tf.Graph().as_default():
    # CREATE DATASET
    if not FLAGS.use_candidate_answers:
      logger.info('Use randomly sampled non answers.')
      train_dataset = make_datasets_w_random_neg_samples(
        data.train_data, test_config.batch_size, test_config.num_non_answers)
      dev_dataset = make_datasets_w_random_neg_samples(
        data.dev_data, test_config.batch_size, test_config.num_non_answers)
      test_dataset = make_datasets_w_random_neg_samples(
        data.test_data, test_config.batch_size, test_config.num_non_answers)
    else:
      logger.info('User non answers from search.')
      train_sample_ids, _, train_candidate_ids = load_candidate_ids(
          os.path.join(query_path, 'train_answer_search_results.pickle'))
      dev_sample_ids, _, dev_candidate_ids = load_candidate_ids(
          os.path.join(query_path, 'dev_answer_search_results.pickle'))
      test_sample_ids, _, test_candidate_ids = load_candidate_ids(
          os.path.join(query_path, 'test_answer_search_results.pickle'))

      train_dataset = make_datasets_w_candidate_samples(
          data.train_data, test_config.batch_size, test_config.num_non_answers, 
          train_sample_ids, train_candidate_ids, sample='top')
      dev_dataset = make_datasets_w_candidate_samples(
          data.dev_data, test_config.batch_size, test_config.num_non_answers, 
          dev_sample_ids, dev_candidate_ids, sample='top')
      test_dataset = make_datasets_w_candidate_samples(
          data.test_data, test_config.batch_size, test_config.num_non_answers, 
          test_sample_ids, test_candidate_ids, sample='top')

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    next_batch = iterator.get_next()
    train_iterator = train_dataset.make_initializable_iterator()
    dev_iterator = dev_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    tf.set_random_seed(FLAGS.tf_random_seed)

    # GLOBAL INITIALIZER
    initializer = tf.random_uniform_initializer(-test_config.init_scale,
                                                test_config.init_scale)

    # QUERY SAVE FILE
    train_query_file, dev_query_file, test_query_file = None, None, None
    if FLAGS.save_query:
      train_query_filename = os.path.join(query_path, "train_question_focus_query_" + datetime_string + ".pickle")
      dev_query_filename = os.path.join(query_path, "dev_question_focus_query_" + datetime_string + ".pickle")
      test_query_filename = os.path.join(query_path, "test_question_focus_query_" + datetime_string + ".pickle")
      train_query_file = open(train_query_filename, 'wb')
      dev_query_file = open(dev_query_filename, 'wb')
      test_query_file = open(test_query_filename, 'wb')

    with tf.name_scope("Test"):
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = Model(next_batch=next_batch,
                  handle=handle,
                  config=test_config,
                  is_training=False,
                  mode=mode,
                  randseed=FLAGS.tf_random_seed)

    sess_config = tf.ConfigProto()
    if FLAGS.allow_gpu_memory_growth:
      sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
      if FLAGS.load_checkpoint == 'focus':
        load_ckpt(sess, m, checkpoint_path_qf_load)
      elif FLAGS.load_checkpoint == 'pairwise':
        load_ckpt(sess, m, checkpoint_path_pw_load)
      elif FLAGS.load_checkpoint == 'semantic':
        load_ckpt(sess, m, checkpoint_path_sm_load)
      elif FLAGS.load_checkpoint == 'ensemble':
        load_ckpt(sess, m, checkpoint_path_en_load)
      else:
        raise ValueError('unknown argument load_checkpoint value {}'.format(FLAGS.load_checkpoint))

      train_handle = sess.run(train_iterator.string_handle())
      dev_handle = sess.run(dev_iterator.string_handle())
      test_handle = sess.run(test_iterator.string_handle())

      # Test on TRAIN set
      if FLAGS.test_on_train_set:
        sess.run(train_iterator.initializer)
        display_sentence = data.display_sentence('train') if FLAGS.print_test_qualitative else None
        get_label = data.get_label('train') if FLAGS.print_test_qualitative else None
        result = run_epoch(sess, m,
                           data_handle=train_handle,
                           display_sentence=display_sentence,
                           # get_label=get_label,
                           query_save_file=train_query_file,
                           is_training=False,
                           eval_rerank=True if FLAGS.use_candidate_answers else False,
                           at_n_list=AT_N_SELFTEST if FLAGS.selftest else AT_N,
                           # minibatch_max_num=(data._train_size / test_config.batch_size + 1)
                           )
        logger.info("Train l2 regu loss: {0:10.5f}".format(result["l2_regu_loss"]))
        logger.info("Train question focus loss: {0:10.5f}".format(result["focus_match_loss"]))
        logger.info("Train pairwise match loss: {0:10.5f}".format(result["pairwise_match_loss"]))
        logger.info("Train semantic match loss: {0:10.5f}".format(result["semantic_match_loss"]))
        # logger.info("Train question focus Prec/Reca/F1: {0:7.5f} / {1:7.5f} / {2:7.5f}".format(
        #     result['question_focus_prec'], result['question_focus_reca'], result["question_focus_f1"]))
        logger.info("Train answer focus    classify accuracy: {0:7.5f}".format(
            result["answer_focus_classify_accuracy"]))
        logger.info("Train answer pairwise classify accuracy: {0:7.5f}".format(
            result["answer_pairwise_classify_accuracy"]))
        logger.info("Train answer semantic classify accuracy: {0:7.5f}".format(
            result["answer_semantic_classify_accuracy"]))
        logger.info("Train answer ensemble classify accuracy: {0:7.5f}".format(
            result["answer_ensemble_classify_accuracy"]))
        if FLAGS.use_candidate_answers:
          show_rerank_results(result)
        logger.info("------------------------------")

      # Test on DEV set
      if FLAGS.test_on_dev_set:
        sess.run(dev_iterator.initializer)
        display_sentence = data.display_sentence('dev') if FLAGS.print_test_qualitative else None
        get_label = data.get_label('dev') if FLAGS.print_test_qualitative else None
        result = run_epoch(sess, m,
                           data_handle=dev_handle,
                           display_sentence=display_sentence,
                           # get_label=get_label,
                           query_save_file=dev_query_file,
                           is_training=False,
                           eval_rerank=True if FLAGS.use_candidate_answers else False,
                           at_n_list=AT_N_SELFTEST if FLAGS.selftest else AT_N)
        logger.info("Valid l2 regu loss: {0:10.5f}".format(result["l2_regu_loss"]))
        logger.info("Valid question focus loss: {0:10.5f}".format(result["focus_match_loss"]))
        logger.info("Valid pairwise match loss: {0:10.5f}".format(result["pairwise_match_loss"]))
        logger.info("Valid semantic match loss: {0:10.5f}".format(result["semantic_match_loss"]))
        # logger.info("Valid question focus Prec/Reca/F1: {0:7.5f} / {1:7.5f} / {2:7.5f}".format(
        #     result['question_focus_prec'], result['question_focus_reca'], result["question_focus_f1"]))
        logger.info("Valid answer focus    classify accuracy: {0:7.5f}".format(
            result["answer_focus_classify_accuracy"]))
        logger.info("Valid answer pairwise classify accuracy: {0:7.5f}".format(
            result["answer_pairwise_classify_accuracy"]))
        logger.info("Valid answer semantic classify accuracy: {0:7.5f}".format(
            result["answer_semantic_classify_accuracy"]))
        logger.info("Valid answer ensemble classify accuracy: {0:7.5f}".format(
            result["answer_ensemble_classify_accuracy"]))
        if FLAGS.use_candidate_answers:
          show_rerank_results(result)
        logger.info("------------------------------")

      # Test on TEST set
      if FLAGS.test_on_test_set:
        sess.run(test_iterator.initializer)
        display_sentence = data.display_sentence('test') if FLAGS.print_test_qualitative else None
        get_label = data.get_label('test') if FLAGS.print_test_qualitative else None
        result = run_epoch(sess, m,
                           data_handle=test_handle,
                           display_sentence=display_sentence,
                           # get_label=get_label,
                           query_save_file=test_query_file,
                           is_training=False,
                           eval_rerank=True if FLAGS.use_candidate_answers else False,
                           at_n_list=AT_N_SELFTEST if FLAGS.selftest else AT_N)
        logger.info("Test l2 regu loss: {0:10.5f}".format(result["l2_regu_loss"]))
        logger.info("Test question focus loss: {0:10.5f}".format(result['focus_match_loss']))
        logger.info("Test pairwise match loss: {0:10.5f}".format(result["pairwise_match_loss"]))
        logger.info("Test semantic match loss: {0:10.5f}".format(result["semantic_match_loss"]))
        # logger.info("Test question focus Prec/Reca/F1: {0:7.5f} / {1:7.5f} / {2:7.5f}".format(
        #     result['question_focus_prec'], result['question_focus_reca'], result["question_focus_f1"]))
        logger.info("Test answer focus    classify accuracy: {0:7.5f}".format(
            result["answer_focus_classify_accuracy"]))
        logger.info("Test answer pairwise classify accuracy: {0:7.5f}".format(
            result["answer_pairwise_classify_accuracy"]))
        logger.info("Test answer semantic classify accuracy: {0:7.5f}".format(
            result["answer_semantic_classify_accuracy"]))
        logger.info("Test answer ensemble classify accuracy: {0:7.5f}".format(
            result["answer_ensemble_classify_accuracy"]))
        if FLAGS.use_candidate_answers:
          show_rerank_results(result)

    if FLAGS.save_query:
      train_query_file.close()
      dev_query_file.close()
      test_query_file.close()


def train():
  np.random.seed(FLAGS.random_seed)
  data = Dataset(max_len=50,
                 num_reviews=0,
                 selftest=FLAGS.selftest,
                 load_review=False)
  train_config = get_config(data)
  logger.info("Train config:" + print_config(train_config))
  mode = []
  if FLAGS.add_focus_loss:
    mode.append('qf')
  if FLAGS.add_pairwise_loss:
    mode.append('pw')
  if FLAGS.add_semantic_loss:
    mode.append('sm')
  if FLAGS.add_ensemble_loss:
    mode.append('en')

  # CONFIGURE PATHS
  query_path = os.path.join('./../intermediate', '{}-query{}'.format(
      FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
  if not os.path.exists(query_path):
    os.makedirs(query_path)

  if not FLAGS.continue_training:
    checkpoint_path_qf_save = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-qf{}'.format(
      FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
    checkpoint_path_pw_save = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-pw{}'.format(
      FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
    checkpoint_path_sm_save = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-sm{}'.format(
      FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
    checkpoint_path_en_save = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-en{}'.format(
      FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
  else:
    checkpoint_path_qf_load = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-qf{}'.format(
      FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
    checkpoint_path_pw_load = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-pw{}'.format(
      FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
    checkpoint_path_sm_load = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-sm{}'.format(
      FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
    checkpoint_path_en_load = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-en{}'.format(
      FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
    checkpoint_path_qf_save = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-qf{}-cont'.format(
      FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
    checkpoint_path_pw_save = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-pw{}-cont'.format(
      FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
    checkpoint_path_sm_save = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-sm{}-cont'.format(
      FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))
    checkpoint_path_en_save = os.path.join(FLAGS.checkpoint_path, '{}-checkpoint-en{}-cont'.format(
      FLAGS.log_file_name, ('-selftest' if FLAGS.selftest else '')))


  with tf.Graph().as_default():

    # CREATE DATASET
    if not FLAGS.use_candidate_answers:
      logger.info('Use randomly sampled non answers.')
      train_dataset = make_datasets_w_random_neg_samples(
          data.train_data, train_config.batch_size, train_config.num_non_answers, shuffle=True)
      dev_dataset = make_datasets_w_random_neg_samples(
          data.dev_data, train_config.batch_size, train_config.num_non_answers)
      test_dataset = make_datasets_w_random_neg_samples(
          data.test_data, train_config.batch_size, train_config.num_non_answers)
    else:
      logger.info('User non answers from search.')
      train_sample_ids, _, train_candidate_ids = load_candidate_ids(
          os.path.join(query_path, 'train_answer_search_results.pickle'))
      dev_sample_ids, _, dev_candidate_ids = load_candidate_ids(
          os.path.join(query_path, 'dev_answer_search_results.pickle'))
      test_sample_ids, _, test_candidate_ids = load_candidate_ids(
          os.path.join(query_path, 'test_answer_search_results.pickle'))

      train_dataset = make_datasets_w_candidate_samples(
          data.train_data, train_config.batch_size, train_config.num_non_answers, 
          train_sample_ids, train_candidate_ids, sample='random', shuffle=True)
      dev_dataset = make_datasets_w_candidate_samples(
          data.dev_data, train_config.batch_size, train_config.num_non_answers, 
          dev_sample_ids, dev_candidate_ids, sample='top')
      test_dataset = make_datasets_w_candidate_samples(
          data.test_data, train_config.batch_size, train_config.num_non_answers, 
          test_sample_ids, test_candidate_ids, sample='top')

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    next_batch = iterator.get_next()
    train_iterator = train_dataset.make_initializable_iterator()
    dev_iterator = dev_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    # SET RANDOM SEED FOR TENSORFLOW
    # FIXME: If set before creating dataset, then the order of data sampling will be the same for each epoch
    tf.set_random_seed(FLAGS.tf_random_seed)

    # GLOBAL INITIALIZER
    initializer = tf.random_uniform_initializer(-train_config.init_scale, train_config.init_scale)
    with tf.name_scope("Train"):
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        embed_initializer = tf.constant_initializer(data.embed_matrix)
        m = Model(next_batch=next_batch,
                  handle=handle,
                  config=train_config,
                  is_training=True,
                  mode=mode,
                  embed_initializer=embed_initializer,
                  randseed=FLAGS.tf_random_seed)

    best_f1_qf, best_result_qf, best_result_pw, best_result_sm, best_result_en = None, None, None, None, None

    sess_config = tf.ConfigProto()
    if FLAGS.allow_gpu_memory_growth:
      sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
      # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
      # tf.set_random_seed(FLAGS.tf_random_seed)  # Set random seed to fixed number
      if FLAGS.continue_training:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if FLAGS.load_checkpoint == 'focus':
          load_ckpt(sess, m, checkpoint_path_qf_load)
        elif FLAGS.load_checkpoint == 'pairwise':
          load_ckpt(sess, m, checkpoint_path_pw_load)
        elif FLAGS.load_checkpoint == 'semantic':
          load_ckpt(sess, m, checkpoint_path_sm_load)
        elif FLAGS.load_checkpoint == 'ensemble':
          load_ckpt(sess, m, checkpoint_path_en_load)
        else:
          raise ValueError('unknown argument load_checkpoint value {}'.format(FLAGS.load_checkpoint))
        m.assign_lr(sess, train_config.learning_rate)
      else:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # sess.run(tf.initialize_all_variables())
        m.assign_lr(sess, train_config.learning_rate)

      train_handle = sess.run(train_iterator.string_handle())
      dev_handle = sess.run(dev_iterator.string_handle())
      test_handle = sess.run(test_iterator.string_handle())

      # OP TO WRITE LOGS TO TENSORBOARD
      if FLAGS.write_summary:
        summary_writer = tf.summary.FileWriter('./tensorboard_summaries', graph=tf.get_default_graph())
      else:
        summary_writer = None

      for i in range(train_config.max_max_epoch):
        # RE-SAMPLE TRAIN DATASETS
        if i > 0 and FLAGS.resample_train_negative:
          start_time = time.time()
          if not FLAGS.use_candidate_answers:
            train_dataset = make_datasets_w_random_neg_samples(
              data.train_data, train_config.batch_size, train_config.num_non_answers, shuffle=True)
          else:
            train_dataset = make_datasets_w_candidate_samples(
              data.train_data, train_config.batch_size, train_config.num_non_answers,
              train_sample_ids, train_candidate_ids, sample='random', shuffle=True)
          train_iterator = train_dataset.make_initializable_iterator()
          train_handle = sess.run(train_iterator.string_handle())
          logger.info("Re-sampled negative samples for training ({} sec)".format(time.time() - start_time))

        logger.info("============================================================")
        logger.info("Epoch %-3d" % (i + 1))
        sess.run(train_iterator.initializer)
        result = run_epoch(sess, m,
                           data_handle=train_handle,
                           is_training=True,
                           summary_writer=summary_writer)
        logger.info("Epoch {0:3d}: Train l2 regu loss: {1:10.5f}".format(i + 1, result["l2_regu_loss"]))
        logger.info("Epoch {0:3d}: Train question focus loss: {1:10.5f}".format(i + 1, result["focus_match_loss"]))
        logger.info("Epoch {0:3d}: Train pairwise match loss: {1:10.5f}".format(i + 1, result["pairwise_match_loss"]))
        logger.info("Epoch {0:3d}: Train semantic match loss: {1:10.5f}".format(i + 1, result["semantic_match_loss"]))
        # logger.info("Epoch {0:3d}: Train question focus Prec/Reca/F1: {1:7.5f} / {2:7.5f} / {3:7.5f}".format(
        #     i + 1, result["question_focus_prec"], result["question_focus_reca"], result["question_focus_f1"]))
        logger.info("Epoch {0:3d}: Train answer focus    classify accuracy: {1:7.5f}".format(
            i + 1, result["answer_focus_classify_accuracy"]))
        logger.info("Epoch {0:3d}: Train answer pairwise classify accuracy: {1:7.5f}".format(
            i + 1, result["answer_pairwise_classify_accuracy"]))
        logger.info("Epoch {0:3d}: Train answer semantic classify accuracy: {1:7.5f}".format(
            i + 1, result["answer_semantic_classify_accuracy"]))
        logger.info("Epoch {0:3d}: Train answer ensemble classify accuracy: {1:7.5f}".format(
            i + 1, result["answer_ensemble_classify_accuracy"]))
        logger.info("------------------------------------------------------------")

        sess.run(dev_iterator.initializer)
        result = run_epoch(sess, m,
                           data_handle=dev_handle,
                           is_training=False)
        logger.info("Epoch {0:3d}: Valid l2 regu loss: {1:10.5f}".format(i + 1, result["l2_regu_loss"]))
        logger.info("Epoch {0:3d}: Valid question focus loss: {1:10.5f}".format(i + 1, result["focus_match_loss"]))
        logger.info("Epoch {0:3d}: Valid pairwise match loss: {1:10.5f}".format(i + 1, result["pairwise_match_loss"]))
        logger.info("Epoch {0:3d}: Valid semantic match loss: {1:10.5f}".format(i + 1, result["semantic_match_loss"]))
        # logger.info("Epoch {0:3d}: Valid question focus Prec/Reca/F1: {1:7.5f} / {2:7.5f} / {3:7.5f}".format(
        #     i + 1, result['question_focus_prec'], result['question_focus_reca'], result["question_focus_f1"]))
        logger.info("Epoch {0:3d}: Valid answer focus    classify accuracy: {1:7.5f}".format(
            i + 1, result["answer_focus_classify_accuracy"]))
        logger.info("Epoch {0:3d}: Valid answer pairwise classify accuracy: {1:7.5f}".format(
            i + 1, result["answer_pairwise_classify_accuracy"]))
        logger.info("Epoch {0:3d}: Valid answer semantic classify accuracy: {1:7.5f}".format(
            i + 1, result["answer_semantic_classify_accuracy"]))
        logger.info("Epoch {0:3d}: Valid answer ensemble classify accuracy: {1:7.5f}".format(
            i + 1, result["answer_ensemble_classify_accuracy"]))

        # Decide wether to save a checkpoint
        eval_f1_qf = result['question_focus_f1']
        eval_result_qf = result['answer_focus_classify_accuracy']
        eval_result_pw = result['answer_pairwise_classify_accuracy']
        eval_result_sm = result['answer_semantic_classify_accuracy']
        eval_result_en = result['answer_ensemble_classify_accuracy']
        better_than = lambda x, y: True if (y is None or x > y) else False
        new_best = False
        if better_than(eval_f1_qf, best_f1_qf):
          best_f1_qf = eval_f1_qf
          # save_ckpt(sess, m, checkpoint_path_qf_save)
          new_best = True
        logger.info("Epoch {0:3d}: Best QF f1 so far: {1:8.5f}{2}".format(
          i + 1, best_f1_qf, ('\t\t\t(NEW!)' if new_best else '')))
        new_best = False
        if better_than(eval_result_qf, best_result_qf):
          best_result_qf = eval_result_qf
          save_ckpt(sess, m, checkpoint_path_qf_save)
          new_best = True
        logger.info("Epoch {0:3d}: Best QF result so far: {1:8.5f}{2}".format(
                    i + 1, best_result_qf, ('\t\t\t(NEW!)' if new_best else '')))
        new_best = False
        if better_than(eval_result_pw, best_result_pw):
          best_result_pw = eval_result_pw
          save_ckpt(sess, m, checkpoint_path_pw_save)
          new_best = True
        logger.info("Epoch {0:3d}: Best PW result so far: {1:8.5f}{2}".format(
                    i + 1, best_result_pw, ('\t\t\t(NEW!)' if new_best else '')))
        new_best=False
        if better_than(eval_result_sm, best_result_sm):
          best_result_sm = eval_result_sm
          save_ckpt(sess, m, checkpoint_path_sm_save)
          new_best = True
        logger.info("Epoch {0:3d}: Best SM result so far: {1:8.5f}{2}".format(
                    i + 1, best_result_sm, ('\t\t\t(NEW!)' if new_best else '')))
        new_best=False
        if better_than(eval_result_en, best_result_en):
          best_result_en = eval_result_en
          save_ckpt(sess, m, checkpoint_path_en_save)
          new_best = True
        logger.info("Epoch {0:3d}: Best EN result so far: {1:8.5f}{2}".format(
                    i + 1, best_result_en, ('\t\t\t(NEW!)' if new_best else '')))
        logger.info("------------------------------------------------------------")

        sess.run(test_iterator.initializer)
        result = run_epoch(sess, m,
                           data_handle=test_handle,
                           is_training=False)
        logger.info("Epoch {0:3d}: Test l2 regu loss: {1:10.5f}".format(i + 1, result["l2_regu_loss"]))
        logger.info("Epoch {0:3d}: Test question focus loss: {1:10.5f}".format(i + 1, result["focus_match_loss"]))
        logger.info("Epoch {0:3d}: Test pairwise match loss: {1:10.5f}".format(i + 1, result["pairwise_match_loss"]))
        logger.info("Epoch {0:3d}: Test semantic match loss: {1:10.5f}".format(i + 1, result["semantic_match_loss"]))
        # logger.info("Epoch {0:3d}: Test question focus Prec/Reca/F1: {1:7.5f} / {2:7.5f} / {3:7.5f}".format(
        #     i + 1, result['question_focus_prec'], result['question_focus_reca'], result["question_focus_f1"]))
        logger.info("Epoch {0:3d}: Test answer focus    classify accuracy: {1:7.5f}".format(
            i + 1, result["answer_focus_classify_accuracy"]))
        logger.info("Epoch {0:3d}: Test answer pairwise classify accuracy: {1:7.5f}".format(
            i + 1, result["answer_pairwise_classify_accuracy"]))
        logger.info("Epoch {0:3d}: Test answer semantic classify accuracy: {1:7.5f}".format(
            i + 1, result["answer_semantic_classify_accuracy"]))
        logger.info("Epoch {0:3d}: Test answer ensemble classify accuracy: {1:7.5f}".format(
            i + 1, result["answer_ensemble_classify_accuracy"]))


def main(_):
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id
  print_flags()
  if FLAGS.train:
    train()
  else:
    test()


if __name__ == "__main__":
  tf.app.run()

