"""ngram baseline solution"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# first tokenize the data
# then extract ngrams
# last do the classcification

import utils
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from nltk import ngrams
from run_classifier_n import ARProcessor
from collections import Counter
from sklearn.linear_model import LinearRegression

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

logger = utils.logger

PAD_IDX = 0
UNK_IDX = 1
MAX_TOKEN_LENGTH = 128
MAX_VAC_SIZE = 50000

def build_vocab(all_tokens, max_vocab_size):
    # Returns:
    # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    # token2id: dictionary where keys represent tokens and corresponding values represent indices
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX 
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token

def tokens2index(tokens, token2id):
        return [token2id[token] if token in token2id else UNK_IDX for token in tokens]

def token2index_dataset(tokens_data, token2id):
    indices_data = []
    for tokens in tokens_data:
        index_list = tokens2index(tokens, token2id)
        indices_data.append(index_list)
    return indices_data

def tokenize_ngram(array,n):
    grams = ngrams(array,n)
    return [' '.join(g) for g in grams]

def tokenize_till_ngram(sent,n = 1):
    tokens = sent.split(' ')
    final_list = [];
    if n == 1:
        final_list = tokens
    else:
        for i in range(1, n+1):
            final_list += tokenize_ngram(tokens, i)
    return final_list

def get_ngram_data(n):
    processor = ARProcessor()

    train_examples = processor.get_train_examples('./data')
    test_examples = processor.get_mytest_examples('./data')
    val_examples = processor.get_dev_examples('./data')

    all_tokens = []
    train_token_dataset = []
    train_labels = []
    val_token_dataset = []
    val_labels = []
    test_token_dataset = []
    test_labels = []
    for sample in train_examples:
        tokens = tokenize_till_ngram(sample.text_a, n = n)
        train_token_dataset.append(tokens)
        train_labels.append(int(sample.label))
        all_tokens += tokens
    for sample in val_examples:
        tokens = tokenize_till_ngram(sample.text_a, n = n)
        val_token_dataset.append(tokens)
        val_labels.append(int(sample.label))
        all_tokens += tokens
    for sample in test_examples:
        tokens = tokenize_till_ngram(sample.text_a, n = n)
        test_token_dataset.append(tokens)
        test_labels.append(int(sample.label))
        all_tokens += tokens
    token2id, id2token = build_vocab(all_tokens, MAX_VAC_SIZE)
    train_data_indices = token2index_dataset(train_token_dataset, token2id)
    val_data_indices = token2index_dataset(val_token_dataset, token2id)
    test_data_indices = token2index_dataset(test_token_dataset, token2id)
    train_data = keras.preprocessing.sequence.pad_sequences(train_data_indices,
                                                            padding='post',
                                                            maxlen=256)
    val_data = keras.preprocessing.sequence.pad_sequences(val_data_indices,
                                                          padding='post',
                                                          maxlen=256)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data_indices,
                                                           padding='post',
                                                           maxlen=256)
    logger.info(test_data[:2])
    logger.info(test_token_dataset[:2])
    logger.info(test_labels[:2])

    print(len(train_data), 'training data')
    print(len(test_data), 'testing data')
    print(len(val_data), 'val data')

    # Change the indx to embeding
    return {'train' : train_data, 'val' : val_data, 'test' : test_data}, \
            {'train' : train_labels, 'val' : val_labels, 'test' : test_labels},\
            token2id, id2token

def convert_embedding(data):
    emblist = []
    for td in data:
        emb = np.zeros(MAX_VAC_SIZE+10)
        emb[td] += 1
        emblist.append(emb)
    return emblist

def run_LR():
    NUM_NGRAM = 2
    data, label, token2id, id2token = get_ngram_data(NUM_NGRAM)
    train_data = convert_embedding(data['train'])
    val_data = convert_embedding(data['val'])
    test_data = convert_embedding(data['test'])

    reg = LinearRegression().fit(train_data, label['train'])
    print(reg.score(val_data, label['val']))
    print(reg.score(test_data, label['test']))


def main():
    logger.info('ngram baseline')

    n=2
    processor = ARProcessor()

    train_examples = processor.get_train_examples('./data')
    test_examples = processor.get_mytest_examples('./data')
    val_examples = processor.get_dev_examples('./data')

    all_tokens = []
    train_token_dataset = []
    train_labels = []
    val_token_dataset = []
    val_labels = []
    test_token_dataset = []
    test_labels = []
    for sample in train_examples:
        tokens = tokenize_till_ngram(sample.text_a, n = n)
        train_token_dataset.append(tokens)
        train_labels.append(int(sample.label))
        all_tokens += tokens
    for sample in val_examples:
        tokens = tokenize_till_ngram(sample.text_a, n = n)
        val_token_dataset.append(tokens)
        val_labels.append(int(sample.label))
        all_tokens += tokens
    for sample in test_examples:
        tokens = tokenize_till_ngram(sample.text_a, n = n)
        test_token_dataset.append(tokens)
        test_labels.append(int(sample.label))
        all_tokens += tokens
    token2id, id2token = build_vocab(all_tokens, MAX_VAC_SIZE)
    train_data_indices = token2index_dataset(train_token_dataset, token2id)
    val_data_indices = token2index_dataset(val_token_dataset, token2id)
    test_data_indices = token2index_dataset(test_token_dataset, token2id)
    train_data = keras.preprocessing.sequence.pad_sequences(train_data_indices,
                                padding='post',
                                maxlen=256)
    val_data = keras.preprocessing.sequence.pad_sequences(val_data_indices,
                                padding='post',
                                maxlen=256)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data_indices,
                                padding='post',
                                maxlen=256)
    logger.info(test_data[:2])
    logger.info(test_token_dataset[:2])
    logger.info(test_labels[:2])
    model = keras.Sequential()
    model.add(keras.layers.Embedding(MAX_VAC_SIZE, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(8, 
        kernel_regularizer=keras.regularizers.l2(0.05), activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.summary()
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_data,
                        train_labels,
                        epochs=100,
                        batch_size=32,
                        validation_data=(val_data, val_labels),
                        verbose=1)
    results = model.evaluate(test_data, test_labels)
    print(results)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def do_scikit():
    all_data = pd.read_csv('./data/test.ft.csv')
    train_df = pd.read_csv('./data/xa100k', header=None)
    val_df = pd.read_csv('./data/xab', header=None)

    train_data = train_df[1]
    train_labels = train_df[0]
    val_data = val_df[1]
    val_labels = val_df[0]
    # tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), max_features=50000)
    tfidf_vect_ngram = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                                            max_features=50000)
    tfidf_vect_ngram.fit(all_data['review'])
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_data)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(val_data)

    reg = LinearRegression().fit(xtrain_tfidf_ngram, train_labels)
    print(xtrain_tfidf_ngram[0])
    print(reg.score(xtrain_tfidf_ngram, train_labels))
    print(reg.score(xvalid_tfidf_ngram, val_labels))

if __name__ == '__main__':
    logger.info('Start')
    #main()
    #run_LR()
    do_scikit()
