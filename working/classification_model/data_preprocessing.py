# ########################################################
# Summary: data preprocessing for training model
# Author: Eunjeong Ahn
# Date of initial creation: 2022.09.05
# Modified date : 2022.09.21
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
import numpy as np


def rm_eng(col):
    """
    remove english in each contents 
    """
    trns1 = col.lower()
    p = re.compile('kicc|kis|kcp|ic|kc|kg|ksnet|koces|jinet|smartro-pg|kicc-pg|kicc-ic|smartro|smartro-if')
    mt_rlst = p.findall(str(trns1))
    if len(mt_rlst) == 1:
        rlst = col.replace(mt_rlst[0], '')
    elif len(mt_rlst) == 2:
        rlst = col.replace(mt_rlst[0], '').replace(mt_rlst[1], '')
    else:
        rlst = col
    return rlst

def trns_eng_to_kor(col):
    """
    transfer from English to Korea in each prononciation
    """
    eng_dic = {'a' : '에이', 'b' : '비', 'c' : '씨', 'd' : '디', 'e' : '이', 'f' : '에프', 'g' : '쥐', 'h' : '에이치', 'i' : '아이', 'j' : '제이', 'k' : '케이', 'l' : '엘', 'm' : '엠', 'n' : '엔', 
           'o' : '오', 'p' : '피', 'q' : '큐', 'r' : '알', 's' : '에스', 't' : '티', 'u' : '유', 'v' : '브이', 'w' : '더블유', 'x' : '엑스', 'y' : '와이', 'z' : '지'}
    trns_wd = col.lower()
    p = re.compile('[A-Za-z]+')
    mt_rlst = p.findall(trns_wd)
    rlst = []
    if len(mt_rlst) > 0:
        for x in mt_rlst:
            tmp = list(x)
            tmp_kor = [eng_dic[y] for y in tmp]
            rlst.append(''.join(tmp_kor))
    return rlst


def rm_keyword(col):
    """
    remove stop words 
    input data : name of store
    output data : name of store removed the stop words
    수정내용 : 존재 할 경우 작성
    """    
    p = re.compile('푸플|배민|우딜')
    mt_rlst = p.findall(str(col))
    if len(mt_rlst) == 1:
        rlst = col.replace(mt_rlst[0], '')
    elif len(mt_rlst) == 2:
        rlst = col.replace(mt_rlst[0], '').replace(mt_rlst[1], '')
    else:
        rlst = col
    return rlst


# 특수문자 제거
def rm_special(col):
    """
    remove special characters
    """
    
    p = re.compile('[0-9]+|[A-Za-z]+|[가-힣]+')
    mt_rlst = p.findall(str(col))
    if len(mt_rlst) > 0:
        rlst = ''.join(mt_rlst)
    else:
        rlst = ''
    return rlst


def mg_token(col1, col2, col3):
    tmp = col1.copy()
    tmp.extend(col2)
    tmp.extend(col3)
    return list(set(tmp))


# execute all methods

def total_preprocessing(data, train_set=False):
    start_time = time.time()
    data['rm_keyword'] = data.apply(lambda x: rm_eng(x['st_name']), axis=1)
    print("[Remove PG keywords process] Runtime: %0.2f Minutes"%((time.time() - start_time)/60))

    start_time = time.time()
    data['trns_keyword'] = data.apply(lambda x: trns_eng_to_kor(x['rm_keyword']), axis=1)
    print("[Transfer English to Korean process] Runtime: %0.2f Minutes"%((time.time() - start_time)/60))

    start_time = time.time()
    data['rm_keyword'] = data.apply(lambda x: rm_keyword(x['rm_keyword']), axis=1)
    print("[Remove order platform keyword process] Runtime: %0.2f Minutes"%((time.time() - start_time)/60))

    start_time = time.time()
    data['rm_keyword1'] = data.apply(lambda x: rm_special(x['rm_keyword']), axis=1)
    print("[Remove Special keyword process] Runtime: %0.2f Minutes"%((time.time() - start_time)/60))


    # tokenizing by blank
    data['split_key'] = data['rm_keyword1'].apply(lambda x: x.split(' '))
    data['split_key1'] = data['split_key'].apply(lambda x: [y for y in x if len(y)<=4])
    data['split_num'] = data.split_key.apply(lambda x: len(x))

    start_time = time.time()
    okt=Okt()
    data['okt_token'] = data['rm_keyword1'].apply(lambda x: okt.nouns(x))
    print("training Runtime: %0.2f Minutes"%((time.time() - start_time)/60))

    data['finl_token'] = data.apply(lambda x: mg_token(x['okt_token'], x['trns_keyword'], x['split_key1']), axis=1)
    if train_set == True:
        rlst_data = data[['st_name', 'finl_token', 'CATEGORY_ID', 'CATEGORY_NM_lv4', 'lv3_new']]
    else:
        rlst_data = data[['st_name', 'finl_token']]
    return rlst_data

