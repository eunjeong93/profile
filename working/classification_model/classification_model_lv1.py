# ########################################################
# Summary : Test code for model develop 
# Author: Eunjeong Ahn
# Date of initial creation: 2022.09.05
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

def train_embedding(data, emb_num, min_num):
    voca_ls = list(set(list(itertools.chain(*list(data['finl_token'])))))
    embedding = FastText(list(data['finl_token']), vector_size = emb_num, window = 12, min_count = min_num, sg = 1)
    embedding.build_vocab(list(data['finl_token']))
    embedding.train(voca_ls, total_examples=len(voca_ls), epochs=300)
    emb_info = {'emb_shape' : embedding.wv.vectors_ngrams.shape,
                'emb_num' : emb_num,
                'emb_weight' : embedding.wv.vectors_ngrams}
    return emb_info

def train_process_y(data, emb_info):
    encoder = LabelEncoder()
    label_y = encoder.fit_transform(list(data['lv3_new']))
    label = encoder.classes_
    oh_y = to_categorical(label_y)
    emb_info['output_n'] = len(data['lv3_new'].unique())
    emb_info['label'] = label
    return oh_y, emb_info

def process_x(data, emb_info, train_set = False):
    if train_set == True:
        token = Tokenizer()
        token.fit_on_texts(data['finl_token'])
        tokened_X = token.texts_to_sequences(data['finl_token'])
        word_dic = token.word_index
        padded = pad_sequences(tokened_X, padding='post')
        
        emb_info['input_n'] = padded.shape[1]
        emb_info['word_dic'] = word_dic
        emb_info['token'] = token
        with open("/root/workspace/local/test_data/m1_meta.pkl", "wb") as fw:
            pickle.dump(emb_info, fw)
    else:
        token = emb_info['token']
        tokened_X = token.texts_to_sequences(data['finl_token'])
        padded = pad_sequences(tokened_X, padding='post', maxlen=emb_info['input_n'])
    return padded, emb_info

def model(emb_shape, emb_num, emb_weight, input_n, output_n):
    model = Sequential()
    model.add(Embedding(emb_shape, emb_num, weights=[emb_weight], input_length = input_n))
    model.add(Conv1D(512, kernel_size = 5, padding = 'same', activation = 'relu'))
    model.add(MaxPool1D(pool_size = 1))
    model.add(LSTM(128, activation = 'tanh', return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, activation = 'tanh', return_sequences = True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(output_n, activation = 'softmax'))
    return model

def training_data(data, emb_num, min_num, train_set = False, batch=128):
    global emb_info
    emb_info = train_embedding(data=data, emb_num=emb_num, min_num=min_num)
    oh_y, emb_info = train_process_y(data=data, emb_info=emb_info)
    padded, emb_info = process_x(data=data, emb_info=emb_info, train_set=train_set)

    training_model = model(emb_shape = emb_info['emb_shape'][0]                        
                        , emb_num = emb_info['emb_num']                        
                        , emb_weight = emb_info['emb_weight']                       
                        , input_n = emb_info['input_n']                        
                        , output_n = emb_info['output_n'])

    training_model.compile(loss = 'categorical_crossentropy', optimizer='adamax', metrics = ['accuracy'])
    checkpoint_path = '/root/workspace/local/model/lv3_my_checkpoint2.ckpt'
    checkpoint = ModelCheckpoint(checkpoint_path, 
                             save_weights_only=True, 
                             save_best_only=True, 
                             monitor='val_accuracy', 
                             verbose=1)

    X_train, X_test, Y_train, Y_test = train_test_split(padded, oh_y, test_size = 0.2, random_state = 100)
    X_test, X_test1, Y_test, Y_test1 = train_test_split(X_test, Y_test, test_size = 0.2, random_state = 100)

    fit_hist1 = training_model.fit(X_train
                                , Y_train
                                , batch_size= batch
                                , epochs=100
                                , validation_data=(X_test, Y_test)
                                , callbacks=[checkpoint])

    training_model.load_weights(checkpoint_path)
    finl_pred = training_model.predict(X_test1)
    finl_pred_ls = [list(x).index(max(x)) for x in finl_pred]
    actual_ls = [list(x).index(max(x)) for x in Y_test1]
    accuracy = np.round(accuracy_score(actual_ls , finl_pred_ls) , 4)

    print(f'소분류 모델의 테스트 정확도는 : {accuracy}')
    return training_model


def model_test(data, train_set=False):
    with open("/root/workspace/local/test_data/m1_meta.pkl", "rb") as fr:
        emb_info = pickle.load(fr)
    checkpoint_path = '/root/workspace/local/model/lv3_my_checkpoint2.ckpt'
    padded, emb_info = process_x(data=data, emb_info = emb_info, train_set=train_set)
    trained_model = model(emb_info['emb_shape'][0]
                        , emb_info['emb_num']
                        , emb_info['emb_weight']
                        , emb_info['input_n']
                        , emb_info['output_n'])
    trained_model.load_weights(checkpoint_path)
    finl_pred = trained_model.predict(padded)
    label = emb_info['label']
    predict_value = []
    for x in finl_pred:
        if max(x) > 0.8:
            prediction = label[list(x).index(max(x))]
        else:
            prediction = "기타"
        predict_value.append(prediction)
    return predict_value
