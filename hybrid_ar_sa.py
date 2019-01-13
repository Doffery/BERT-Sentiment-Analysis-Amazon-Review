'''Trying combine sentense embedding from bert with ngram features'''

import ngram_baseline
import numpy as np
import pandas as pd
import json
import lightgbm as lgb
from computeBERTSentenceEmbedding import genSentenceEmbedding
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

ALL_DATA_FILE = './data/test.ft.csv'
TRAIN_DATA_FILE = './data/xa10k'
VAL_DATA_FILE = './data/xab'
TEST_DATA_FILE = './data/xaa'
EMBEDDING_TRIAN_FILE = TRAIN_DATA_FILE+".jsont4"
EMBEDDING_VAL_FILE = VAL_DATA_FILE+".jsont4"
EMBEDDING_TEST_FILE = TEST_DATA_FILE+".jsonl"
EMB_MODE = "CLS"
NUM_LAYERS = 4
NUM_NGRAM = 2
NGRAM_VAL_LENGTH = 50005

def combine_emmedding_ngram(embedding_file, ngramdata):
    print('Combing', embedding_file, NUM_NGRAM, 'ngram data')
    print(type(ngramdata))
    # return ngramdata
    embedding_data = []
    with open(embedding_file, encoding='utf-8') as data_file:
        # data_file.readline()  # skip the first line
        i = 0
        for line in data_file.readlines() :
            data = json.loads(line)
            vecSent,tokenizedSent,index = genSentenceEmbedding(data,
                                                               EMB_MODE,
                                                               NUM_LAYERS)
            # print(vecSent)
            # print(len(vecSent))
            # print(tokenizedSent)
            # _id = index.split(';;')[0]
            # label = index.split(';;')[1]
            # assert(len(vecSent) == 768)
            # assert(ngramdata[i].shape == (1,50000))
            cont_feature = np.concatenate([vecSent, 
                ngramdata[i].toarray().flatten()])
            embedding_data.append(cont_feature)
            i += 1
    return embedding_data

if __name__ == "__main__":
    all_data = pd.read_csv(ALL_DATA_FILE)
    train_df = pd.read_csv(TRAIN_DATA_FILE, header=None)
    val_df = pd.read_csv(VAL_DATA_FILE, header=None)

    val_df = val_df.iloc[1:]
    #train_df = train_df.iloc[1:]

    train_text = train_df[1]
    train_labels = train_df[0]
    val_text = val_df[1]
    val_labels = val_df[0]
    print('positive val C:', sum(val_labels))
    # tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), max_features=50000)
    tfidf_vect_ngram = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                                       ngram_range=(1, 2), max_features=50000,
                                       binary=True)
    tfidf_vect_ngram.fit(all_data['review'])
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_text)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(val_text)
    print('tngram.shape', xtrain_tfidf_ngram.shape)
    print('vngram.shape', xvalid_tfidf_ngram.shape)
    train_data = combine_emmedding_ngram(EMBEDDING_TRIAN_FILE,
            xtrain_tfidf_ngram)
    print('train_data shape', np.array(train_data).shape)
    val_data = combine_emmedding_ngram(EMBEDDING_VAL_FILE,
            xvalid_tfidf_ngram)
    print('val_data shape', np.array(val_data).shape)
    # test_data = combine_emmedding_ngram(EMBEDDING_TEST_FILE, data['test'], token2id)

    xtrain_tfidf_ngram = xtrain_tfidf_ngram.astype(np.float32, copy=False)
    xvalid_tfidf_ngram = xvalid_tfidf_ngram.astype(np.float32, copy=False)
    gbm = lgb.LGBMClassifier(num_leaves=51,
            learning_rate=0.05,
            n_estimators=150)
    gbm.fit(xtrain_tfidf_ngram, train_labels,
            eval_set=[(xvalid_tfidf_ngram, val_labels)])
    pre = gbm.predict(xvalid_tfidf_ngram)
    print(accuracy_score(pre, val_labels))

    # reg = LogisticRegression(solver='liblinear', C=0.1).fit(train_data, train_labels)
    # reg_ngram = LogisticRegression(solver='liblinear', C=0.2).fit(
    #         xtrain_tfidf_ngram, train_labels)
    # print(reg.score(train_data, train_labels))
    # print(reg.score(val_data, val_labels))
    # r1 = reg.predict(val_data)
    # r2 = reg_ngram.predict(xvalid_tfidf_ngram)
    # nt_index = (r2 == val_labels) * (r1 != val_labels)
    # print(sum(nt_index))
    # val_df[nt_index].to_csv('./tmp/ngram_true_emb_false', sep='')
