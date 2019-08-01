# product-qa

## Requirement

**IMPORTANT**: Set required environment variables by calling`. set_envs.sh` from this project's home directory.

It basically does the following:
```bash
export PQA_HOME=$PWD
export PQA_DOMAIN=baby
export PQA_DATA_SOURCE=${PQA_HOME}/preprocess/${PQA_DOMAIN}_qa_review_spacy_preprocessed.txt
export PQA_DATA_PATH=${PQA_HOME}/preprocess/${PQA_DOMAIN}
export PQA_GLOVE_DIM=300
export PQA_GLOVE_PATH=${PQA_HOME}/data-raw/glove/glove.42B.300d.txt
```

## Dataset
The original dataset are from [this work](https://arxiv.org/abs/1512.06863), and the datasets can be downloaded here: 
[qa](http://jmcauley.ucsd.edu/data/amazon/qa/) and [reviews](http://jmcauley.ucsd.edu/data/amazon/).

To preprocess the data, run the following commands:
```bash
cd ${PQA_HOME}/preprocess/scripts
python preprocess_from_raw_data.py
python preprocess_with_spacy.py
```

## Neural Network Model
### Train
```bash
cd ${PQA_HOME}/src
python run.py --log_file_name=exp
```
This will save tensorflow model checkpoints to folder `${PQA_HOME}/checkpoints/exp-checkpoint-<xx>` and logs to 
`${PQA_HOME}/logs/train_exp_<xxxxxxx_xxxxxx>.log`.

### Test
```bash
cd ${PQA_HOME}/src
python run.py --log_file_name=exp --train=false --save_query
```
Set `save_query` argument to save the re-weighted query words to 
`${PQA_HOME}/intermediate/exp-query/[train|dev|test]_quesiton_focus_query_xxxxxxxx_xxxxxx.pickle` and logs to 
`${PQA_HOME}/logs/test_exp_<xxxxxxx_xxxxxx>.log`.

## Evaluate

### Word similarties

#### Word similarity based on trained word embeddings
```bash
cd ${PQA_HOME}/src
python eval_word_expansion.py --log_file_name=exp
```
This will produce:
* word similarities: `${PQA_HOME}/intermediate/exp-query/word_similarity_rnn_input_new.pickle`
* word similarities: `${PQA_HOME}/intermediate/exp-query/word_similarity_rnn_output_new.pickle`
* qualitative results: `${PQA_HOME}/result/exp/temp_results_word_expansion_rnn_input.txt`
* qualitative results: `${PQA_HOME}/result/exp/temp_results_word_expansion_rnn_output.txt`
* intermediate data: `${PQA_HOME}/intermediate/exp-query/word_embed_rnn_output.pickle`

#### Word similarity based on trained model
```bash
cd ${PQA_HOME}/src
python eval_sent_level_expansion.py --log_file_name=exp
```
This will produce:
* `${PQA_HOME}/intermediate/exp-query/[dev|test]_question_focus_query_expansion_xxxxxxxx_xxxxxx.pickle`

