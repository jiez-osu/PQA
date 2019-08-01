import pdb
import inspect

class Config(object):
  """Train config."""
  init_scale = 0.001
  learning_rate = 0.0001
  lr_decay = 0.999
  w_embed_size = 0
  hidden_size = 300
  max_epoch = 10
  max_max_epoch = 100
  keep_prob = 1.0
  word_embed_keep_prob = 1.0
  batch_size = 64
  vocab_size = 0
  l2_regu_weight = 1e-5
  embed_mode = "glove_init"
  ans_size = 1
  doc_size = 0
  n_cntxt_w = 100
  max_len = 0
  num_reviews=20
  num_non_answers=5
  rnn_hidden_size=128
  focus_hidden_size=128


class TestConfig(object):
  """Test config."""
  init_scale = 0.001
  learning_rate = 0.0001
  lr_decay = 0.999
  w_embed_size = 0
  hidden_size = 300
  max_epoch = 10
  max_max_epoch = 100
  keep_prob = 1.0
  word_embed_keep_prob = 1.0
  batch_size = 128
  vocab_size = 0
  l2_regu_weight = 1e-4
  embed_mode = "glove_init"
  ans_size = 1
  doc_size = 0
  n_cntxt_w = 100
  max_len = 0
  num_reviews=20
  num_non_answers=5
  rnn_hidden_size=128
  focus_hidden_size=128


class SelfTestConfig(object):
  """self test config."""
  init_scale = 0.001
  learning_rate = 0.01
  lr_decay = 0.999
  max_grad_norm = 5
  w_embed_size = 0
  hidden_size = 300
  max_epoch = 5
  max_max_epoch = 11 
  keep_prob = 1.0
  word_embed_keep_prob = 1.0
  batch_size = 6
  vocab_size = 0
  l2_regu_weight = 0.
  embed_mode = "glove_init"
  ans_size = 1
  doc_size = 0
  n_cntxt_w = 100
  max_len = 0
  num_reviews=20
  num_non_answers=4
  rnn_hidden_size=128
  focus_hidden_size=128


def print_config(config):
  rslt =""
  for attr, val in inspect.getmembers(config):
    if not (attr.startswith("__") and attr.endswith("__")):
      rslt += "\n\t\t{0:<24} : {1}".format(attr, val)
  return rslt


if __name__ == "__main__":
  config = Config()
  print print_config(config)
  config = TestConfig()
  print print_config(config)
  config = SelfTestConfig()
  print print_config(config)
