# Riker

## 1 Introduction
This repository contains source code and datasets for paper "Riker: Mining Rich Keyword Representations for Interpretable Product Question Answering" (SIGKDD 2019). 


## 2 Dataset
The original dataset are from [this work](https://arxiv.org/abs/1512.06863). Both the QA and review datasets can be downloaded following the instructions in this paper.

The preprocessed QA data can be found [here](https://drive.google.com/drive/folders/1G1t3ifTZcZ11G5x8XW7zXtgT3tBbhJfo?usp=sharing). The preprocessed review datasets are not uploaded due to their sizes, but they can be preprocessed from the raw data using the preprocessing scripts.

To preprocess the data, run the following commands:
```bash
cd ${PQA_HOME}/preprocess/scripts
python preprocess_from_raw_data.py
python preprocess_with_spacy.py
```

## 3 Code

### Requirement

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

### Neural Network Model
#### Train
```bash
cd ${PQA_HOME}/src
python run.py --log_file_name=exp
```
This will save tensorflow model checkpoints to folder `${PQA_HOME}/checkpoints/exp-checkpoint-<xx>` and logs to 
`${PQA_HOME}/logs/train_exp_<xxxxxxx_xxxxxx>.log`.

#### Test
```bash
cd ${PQA_HOME}/src
python run.py --log_file_name=exp --train=false --save_query
```
Set `save_query` argument to save the re-weighted query words to 
`${PQA_HOME}/intermediate/exp-query/[train|dev|test]_quesiton_focus_query_xxxxxxxx_xxxxxx.pickle` and logs to 
`${PQA_HOME}/logs/test_exp_<xxxxxxx_xxxxxx>.log`.

### Evaluate

#### Word similarties

##### Word similarity based on trained word embeddings
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

##### Word similarity based on trained model
```bash
cd ${PQA_HOME}/src
python eval_sent_level_expansion.py --log_file_name=exp
```
This will produce:
* `${PQA_HOME}/intermediate/exp-query/[dev|test]_question_focus_query_expansion_xxxxxxxx_xxxxxx.pickle`

## 4 Citation
Please kindly cite our paper if you use the code or the datasets in this repo:
```
@inproceedings{Zhao:2019:RMR:3292500.3330985,
 author = {Zhao, Jie and Guan, Ziyu and Sun, Huan},
 title = {Riker: Mining Rich Keyword Representations for Interpretable Product Question Answering},
 booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
 series = {KDD '19},
 year = {2019},
 isbn = {978-1-4503-6201-6},
 location = {Anchorage, AK, USA},
 pages = {1389--1398},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3292500.3330985},
 doi = {10.1145/3292500.3330985},
 acmid = {3330985},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {interpretable search, product qa, question representation},
} 
```


