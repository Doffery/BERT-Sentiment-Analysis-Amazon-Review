"""ngram baseline solution"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# first tokenize the data
# then extract ngrams
# last do the classcification

import utils
import os
import tensorflow as tf
from tensorflow import keras
from nltk import ngrams
from run_classifier_n import ARProcessor
from collections import Counter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

def token2index_dataset(tokens_data, token2id):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[token] if token in token2id else UNK_IDX for token in tokens]
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



if __name__ == '__main__':
    logger.info('Start')
    main()
