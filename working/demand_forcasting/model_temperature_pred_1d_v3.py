"""
demand forcasting for 1 day
"""
#%%
#=================================
# 1. Import module
#=================================

# Import basic modules
import os
import sys
import logging
from datetime import timedelta, date, datetime
import numpy as np
import pandas as pd

# Import modules for visualizing
import matplotlib.pyplot as plt
import seaborn as sns

# Import modules for models
import category_encoders as ce
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from mlxtend.regressor import StackingCVRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

# 라이브러리 간 충돌 방지
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# custom module
from common.batch.learner import Learner
from common.io.read import main_read
from common.util.decorator import timer
from common.util.aargparser import get_aargparser
from common.feature.feature import FeatureEngineering as fe
from src.feature_engineering.fe_temperature_pred_1d_v3 import DailyTemperatureRatioFeature

# logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(
    stream=sys.stdout,
    format='[%(asctime)s] %(levelname)s : %(message)s')

#=================================
# 2. Define Class
#=================================

class DailyTemperatureRatioPrediction(Learner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_date = self._date_parse('2021-05-01')
        self.target = kwargs["target"]

        self.table_name = "table_name"
        self.save_idx = ["column_list"]

        self.feature = DailyTemperatureRatioFeature(**kwargs)

    def filename_for_save(self):
        return f"{self.table_name}_{self.target}"
    
    def get_data(self):

        try:
            feature_file_path = self.feature.get_s3_file_path()
            df = main_read.read_s3_file(feature_file_path)
        except Exception as e:
            self.feature.process()
            df = self.feature.result['df']
        df = df.drop(columns=['update_dt'])
        df['date'] = pd.to_datetime(df['date'])
        return df

    def get_target(self, df):

        # target value - log transform
        # remove contents because of security
        return df

    def add_rolling_median(self, df):
        """ past same weekday rolling variable """

        ## recent 4 weeks same weekday average lag_feature
        index_cols = ['column_list']
        df_group = df.groupby(index_cols)['target']
        df['column'] = fe.rolling_median(df_group, window=4, shift_size=1)

        ## recent 4 days weekend average lag_feature
        index_cols = ['column_list']
        df_group = df.groupby(index_cols)['target']
        df['column'] = fe.rolling_median(df_group, window=4, shift_size=1)

        ## recent 4 days promotion average lag_feature
        index_cols = ['column_list']
        df_group = df.groupby(index_cols)['target']
        df['column'] = fe.rolling_median(df_group, window=4, shift_size=1)

        ## null value  
        cols = ['column_list']
        for c in cols:
            n = c + '_temp'
            temp = df.copy()
            temp = temp.groupby(['column_list'])[c].median().reset_index().rename(columns={c:n})
            df = df.merge(temp, how='left',on=['column_list'])
            df[c] = np.where(df[c].isna() == True, df[n], df[c])
            df = df.drop(columns=[n])

        return df

    def add_temp_target(self, df):
        """ create temporary future target
            remove content because of security
        """

        return df

    def add_ewma(self, df):
        """ 지수 가중 평균 """

        ## 파라미터 설정 
        params = [
            # weight for past time
                # remove area -> for security
            # weight for current time
                # remove area -> for security
 
        ]
        param = { 'column': 0.7, 'column': 0.7, 'column': 0.7, 'column': 0.7}
        # default_value
        for p in params:
            if self.target_date >= self._date_parse(p['column']) and \
                    self.target_date <= self._date_parse(p['column']):
                param = p 
                break

        df['column'] = \
            df.groupby(['column_list'])[
                'target'].transform(
                lambda x: x.ewm(alpha=param['column'], adjust=True).mean())
        df['column'] = df.groupby(['column_list'])['target'].transform(
            lambda x: x.ewm(alpha=param['ewma_1w'], adjust=True).mean())
        df['column'] = df.groupby(['column_list'])['target'].transform(
            lambda x: x.ewm(alpha=param['ewma_1d'], adjust=True).mean())
        df['column'] = \
            df.groupby(['column_list'])['target'].transform(
                lambda x: x.ewm(alpha=param['column'], adjust=True).mean())
        return df     
    
    def train_test_split(self, df):
        """
        train_test split and Scaling
        """

        # drop useless column
        df = df.drop(['column_list'], axis=1)
        df = df.dropna()
        df = df.replace([np.inf, -np.inf], 0)
        df = df.sort_values(by=['column_list'])
        df = df.reset_index(drop=True)

        # get the index after which test set starts
        test_start_date = self.target_date.to_date
        df = df.set_index(df['date'])
        df.index = pd.to_datetime(df.index)
        df.pop('date')
        X_train = df.drop(['target', 'column_list'], axis=1).loc[:pd.to_datetime(test_start_date) - timedelta(days=1)]
        y_train = df['target'].loc[:pd.to_datetime(test_start_date) - timedelta(days=1)]
        X_test = df.drop('target', axis=1).loc[test_start_date:]
        y_test = df['target'].loc[test_start_date:]
        pred_df = X_test[['ccolumn_list']].copy()
        X_test = X_test.drop(['column_list'], axis=1)

        # print(X_test['features_combination'].unique())

        # process for categorical variables
        cat_cols = ['column_list']
        tenc = ce.TargetEncoder(cols=cat_cols, smoothing=0.7)
        X_train = tenc.fit_transform(X_train, y_train)
        X_test = tenc.transform(X_test)

        # Scaling
        scaler = MinMaxScaler()
        X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
        X_test[X_train.columns] = scaler.transform(X_test[X_train.columns])

        return X_train, X_test, y_train, y_test, pred_df

    def define_models(self):
   
        # Light Gradient Boosting Regressor
        lightgbm = LGBMRegressor(objective='regression',
                                 num_leaves=20,
                                 learning_rate=0.01,
                                 n_estimators=7000,
                                 max_bin=200,
                                 bagging_seed=8,
                                 feature_fraction_seed=8,
                                 verbose=-1,
                                 random_state=42)

        # Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=1200,
                                   max_depth=16,
                                   min_samples_split=5,
                                   min_samples_leaf=5,
                                   max_features=None,
                                   oob_score=True,
                                   random_state=42)

        # MLP Regressor
        mlp = MLPRegressor(
            hidden_layer_sizes=(16, 8),
            max_iter=1000,
            solver='adam',
            alpha=1e-05,
            batch_size=16,
            activation='relu',
            shuffle=False,
            early_stopping=True,
            learning_rate='constant',
            tol=0.0001,
            verbose=True,
            validation_fraction=0.15
        )

        # MLP Regressor
        mlp2 = MLPRegressor(
            hidden_layer_sizes=(16, 8),
            max_iter=1000,
            solver='adam',
            alpha=1e-05,
            batch_size=16,
            activation='relu',
            shuffle=True,
            early_stopping=True,
            learning_rate='constant',
            tol=0.0001,
            verbose=True,
            validation_fraction=0.15
        )

        # create stacking model
        stack_gen = StackingCVRegressor(regressors=(lightgbm, rf, mlp, mlp2),
                                        meta_regressor=lightgbm,
                                        use_features_in_secondary=True)
        return stack_gen

    def modeling(self, df):
        X_train, X_test, y_train, y_test, pred_dataframe = self.train_test_split(df)
        logging.info("split train test")

        stack_gen = self.define_models()
        stacking_model = stack_gen.fit(np.array(X_train), np.array(y_train))
        pred = np.expm1(stacking_model.predict(np.array(X_test)))
        pred_dataframe = pred_dataframe.reset_index()
        pred_dataframe['type'] = self.target
        pred_dataframe['target'] = pred 
        pred_dataframe['target_org'] = pred
        pred_dataframe['model_pred_date'] = date.today()
        
        return pred_dataframe

    def preprocess(self,**kwargs):
        """ import data """
        df = self.get_data()
        print('---------------- center_dlvy ----------------')
        print(df[df['biz_ymd'] == self.target_date][['column_list']].drop_duplicates())
        logging.info(df)
        
        """ preprocessing """
        df = self.get_target(df) # preprocessing
        logging.info(df)

        print('----------------lag feature----------------')
        df = self.add_rolling_median(df) 
         
        print('----------------temp_target----------------')
        df = self.add_temp_target(df) 
        
        print('---------------- ewma ----------------')
        df = self.add_ewma(df) 
        
        print('---------------- end preprocess ----------------')
        logging.info(df)

        return {'df': df}

    def learn_fn(self, df):
        """ train model and inference """
        pred_dataframe = self.modeling(df)
        logging.info(pred_dataframe)
        return pred_dataframe

    ## additional logic for model's performance -> remove
    def postprocess(self, **kwargs):
        """
        remove contents because of security
        """
        return df

if __name__ == "__main__":
    parser = get_aargparser()
    parser.add_argument('--target', '-ta',
                        type=str,
                        default='type',
                        choices=['type of training data'],
                        help="Format example) --target : {type of training data list}")
    args = parser.parse_args()
    prediction = DailyTemperatureRatioPrediction(**args.__dict__)
    pred = prediction.process()
