# ########################################################
# Summary: Test code for model develop level 2
# Author: Eunjeong Ahn
# Date of initial creation: 2022.09.17
# Version : version 0.1
# #########################################################

import pandas as pd
import re
import time
import requests
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from konlpy.tag import Okt
from konlpy.tag import Kkma
from konlpy.tag import Hannanum
from konlpy.tag import Komoran
from gensim.models import FastText
import itertools
from gensim.models import word2vec
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import *
import numpy as np

def create_meta_info(dat):
    cate = []
    idx = []
    emb_shape = []
    emb_weight = []
    pad_len = []
    path = []
    out_len = []
    label_ls = []
    token_ls = []
    lv3_ls = list(dat['lv3_new'].unique())
    # training data information
    for lb in tqdm(lv3_ls):
        cate.append(lb)
        # with open(f"/root/workspace/local/test_data/{lb}_test_data_index.pkl", "rb") as fr:
        #     idx_info = pickle.load(fr)
        X_train, X_test, Y_train, Y_test = train_test_split(
            dat.loc[dat.lv3_new == lb]['finl_token'], dat.loc[dat.lv3_new == lb]['CATEGORY_ID'], test_size=0.1, random_state=100)
        X_test, X_test1, Y_test, Y_test1 = train_test_split(X_test, Y_test, test_size=0.2, random_state=100)

        train_idx = list(X_train.index)
        train_idx.extend(list(X_test1.index))
        test_idx = list(X_test.index)
        idx_info = (train_idx, test_idx)

        idx.append(idx_info)
        path.append(f'/root/workspace/local/model/{lb}_my_checkpoint2.ckpt')

        new_dat = dat.loc[dat.index.isin(idx_info[0]), :]

        encoder = LabelEncoder()
        token = Tokenizer()

        label_y = encoder.fit_transform(list(new_dat.CATEGORY_NM_lv4))
        label = encoder.classes_

        voca_ls = list(set(list(itertools.chain(*list(new_dat.finl_token)))))
        emb_num = 100
        embedding = FastText(list(new_dat.finl_token),
                            vector_size=emb_num, window=5, min_count=20, sg=1)
        embedding.build_vocab(list(new_dat.finl_token))
        embedding.train(voca_ls, total_examples=len(voca_ls), epochs=300)

        token.fit_on_texts(new_dat.finl_token)
        tokened_X = token.texts_to_sequences(new_dat.finl_token)

        word_dic = token.word_index
        padded = pad_sequences(tokened_X, padding='post')

        emb_shape.append(embedding.wv.vectors_ngrams.shape)
        emb_weight.append(embedding.wv.vectors_ngrams)
        pad_len.append(padded.shape[1])
        out_len.append(len(new_dat.CATEGORY_NM_lv4.unique()))
        label_ls.append(label)
        token_ls.append(token)

    meta_info = pd.DataFrame({'cate': cate
                            , 'idx': idx
                            , 'path': path
                            , 'emb_shape': emb_shape
                            , 'emb_weight': emb_weight
                            , 'pad_len': pad_len
                            , 'out_len': out_len
                            , 'label_ls': label_ls
                            , 'token': token_ls})
    with open("/root/workspace/local/test_data/meta.pkl", "wb") as fw:
        pickle.dump(meta_info, fw)
    return 

def model_create(voca_shape, emb_num, emb_weight, input_len, output_len):

    model1 = Sequential()
    model1.add(Embedding(voca_shape, emb_num, weights=[emb_weight], input_length=input_len))
    # Embedding layer for test 
    # model1.add(Embedding(voca_shape, emb_num, input_length=input_len))
    model1.add(Conv1D(512, kernel_size=5, padding='same', activation='relu'))
    model1.add(MaxPool1D(pool_size=1))
    model1.add(LSTM(128, activation='tanh', return_sequences=True))
    model1.add(Dropout(0.2))
    model1.add(LSTM(64, activation='tanh', return_sequences=True))
    model1.add(Dropout(0.2))
    model1.add(Flatten())
    model1.add(Dense(128, activation='relu'))
    model1.add(Dense(output_len, activation='softmax'))

    return model1


def model_training(dat, category_label, emb_num=100, min_count=20, batch_size=128, epoch=100):
    """
    Training model for particular category
    input values
        dat : data including all categories
        category_label : target category for training
        emb_num : hyperparameter in Fasttext algorithm
        min_count : hyperparameter in Fasttext algorithm
        batch_size : batch_size when training model
        epoch : epoch
    output values
        model weight : save by path for each category
        model : for applying test data 
    other glonle variable (used in other method)
        idx : target category's index
        pad_num : input variable demension(length)
        label : list of target (y)
    """
    global cate_idx, pad_num, label, checkpoint_path, token1

    with open("/root/workspace/local/test_data/meta.pkl", "rb") as fr:
        meta_info = pickle.load(fr)
    i = meta_info[meta_info.cate == category_label].index[0]
    cate_idx = meta_info.idx[i]

    new_dat = dat.loc[dat.index.isin(cate_idx[0]), :]
    encoder = LabelEncoder()
    token = Tokenizer()
   
    voca_ls = list(set(list(itertools.chain(*list(new_dat.finl_token)))))
    emb_num = emb_num
    voca_size = len(voca_ls)
    embedding = FastText(list(new_dat.finl_token),
                         vector_size=emb_num, window=5, min_count=min_count, sg=1)
    embedding.build_vocab(list(new_dat.finl_token))
    embedding.train(voca_ls, total_examples=len(voca_ls), epochs=300)

    label_y = encoder.fit_transform(list(new_dat.CATEGORY_NM_lv4))
    label = encoder.classes_
    oh_y = to_categorical(label_y)

    token.fit_on_texts(new_dat.finl_token)
    token1 = token
    tokened_X = token.texts_to_sequences(new_dat.finl_token)

    word_dic = token.word_index
    padded = pad_sequences(tokened_X, padding='post')

    X_train, X_test, Y_train, Y_test = train_test_split(
        padded, oh_y, test_size=0.1, random_state=100)
    X_test, X_test1, Y_test, Y_test1 = train_test_split(
        X_test, Y_test, test_size=0.2, random_state=100)

    pad_num = padded.shape[1]

    model1 = model_create(
                            voca_shape=embedding.wv.vectors_ngrams.shape[0]    # fasttext 사용할때 활성화
                            # voca_shape = voca_size     # 일반 embedding 사용할때 활성화
                            , emb_num=emb_num
                            , emb_weight=embedding.wv.vectors_ngrams
                            , input_len=pad_num
                            , output_len=len(new_dat.CATEGORY_NM_lv4.unique())
                         )

    model1.compile(loss='categorical_crossentropy',
                   optimizer='adamax', metrics=['accuracy'])

    checkpoint_path = f'/root/workspace/local/model/{category_label}_my_checkpoint2.ckpt'

    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_accuracy',
                                 verbose=1)

    fit_hist1 = model1.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch, validation_data=(
        X_test, Y_test), callbacks=[checkpoint])

    finl_pred = model1.predict(X_test1)
    finl_pred_ls = [list(x).index(max(x)) for x in finl_pred]
    actual_ls = [list(x).index(max(x)) for x in Y_test1]
    accuracy = np.round(accuracy_score(actual_ls, finl_pred_ls), 4)

    meta_info.emb_shape[i] = embedding.wv.vectors_ngrams.shape
    meta_info.emb_weight[i] = embedding.wv.vectors_ngrams

    with open("/root/workspace/local/test_data/meta.pkl", "wb") as fw:
        pickle.dump(meta_info, fw)

    print(f'소분류 {category_label}의 세분류 모델 Training 데이터셋 정확도 : {accuracy}')

    del model1
    return 
    # return meta_info, model1


def model_test(dat, category_label, train_set=False, emb_num=100):
    """
    inference
    input values
        dat : data including all categories
        meta_info : information about inference (including index, lebel, input size etc)
        category_label : target category
    output value
        data including category level2
    """
    with open("/root/workspace/local/test_data/meta.pkl", "rb") as fr:
        meta_info = pickle.load(fr)

    i = meta_info[meta_info.cate == category_label].index[0]
    path = f'/root/workspace/local/model/{category_label}_my_checkpoint2.ckpt'
    if train_set == True:
        new_idx = meta_info['idx'][i]
        new_dat = dat.loc[dat.index.isin(new_idx[1]), :]
    else:
        new_dat = dat.loc[dat['lv3_new'] == category_label]
    encoder = LabelEncoder()
    token = meta_info.token[i]
    # voca_ls = list(set(list(itertools.chain(*list(dat.loc[dat.index.isin(idx[0]), :].finl_token)))))
    emb_num = emb_num

    label = meta_info['label_ls'][i]
    # label_y = encoder.fit_transform(list(new_dat.CATEGORY_NM_lv4))

    tokened_X = token.texts_to_sequences(new_dat.finl_token)

    word_dic = token.word_index
    padded = pad_sequences(tokened_X
                        , padding='post'
                        , maxlen=meta_info['pad_len'][i])

    model1 = model_create(
                            voca_shape=meta_info['emb_shape'][i][0]
                            , emb_num=emb_num
                            , emb_weight=meta_info['emb_weight'][i]
                            , input_len=meta_info['pad_len'][i]
                            , output_len=meta_info['out_len'][i]
                        )

    model1.load_weights(path)
    finl_pred = model1.predict(padded)
    finl_pred_ls = [list(x).index(max(x)) for x in finl_pred]
    expected_value = [label[x] for x in finl_pred_ls]

    return expected_value, list(new_dat.index)
