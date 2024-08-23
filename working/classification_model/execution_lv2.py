import data_preprocessing as p1
import classification_model_lv1 as m1
import classification_model_lv2 as m2
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys


with open("/root/workspace/local/test_data/data_for_train.pkl", "rb") as fr:
    data = pickle.load(fr)


# 너무 적은 카테고리 일단 제외
data = data.loc[data.CATEGORY_LV3_ID != '20617']
data = p1.total_preprocessing(data, train_set=True)
lv3_ls = data.CATEGORY_LV3_ID.unique()

def execute(data, cate, min_cnt):
    start_time = time.time()
    try:
        m2.model_training(data, category_label = cate, min_count=min_cnt, emb_num=100, batch_size=128, epoch=100)
    except:
        m2.create_meta_info(data)
        m2.model_training(data, category_label = cate, min_count=min_cnt, emb_num=100, batch_size=128, epoch=100)
    print("training Runtime: %0.2f Minutes"%((time.time() - start_time)/60))

def main():
    ty = sys.argv[1]
    min_cnt = int(sys.argv[2])
    if ty == 'all':
        for cate in tqdm(lv3_ls):
            m2.model_training(data, cate, min_count=min_cnt)
    elif ty == 'cate':
        cate = sys.argv[3]
        m2.model_training(data, cate, min_count=min_cnt)
    print('--- Finish Training ---')
    m2.create_dictionary(data)
    print('--- Finish to create dictionary --')

if __name__ == '__main__':
    main()