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

def main():
    start_time = time.time()
    m1.training_data(data, emb_num = e_num, min_num = m_num, train_set=True)
    print("training Runtime: %0.2f Minutes"%((time.time() - start_time)/60))

if __name__ == '__main__':
    e_num = sys.argv[1]
    m_num = sys.argv[2]
    main()