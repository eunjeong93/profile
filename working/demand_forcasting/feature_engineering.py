
#%%
#=================================
# 1. Import module
#=================================

# Import basic modules
import os
import sys
import logging
from datetime import timedelta
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

# avoid collision between libraries
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# custom module
from common.batch.dataprocessor import DataProcessor
from common.io.read import main_read
from common.util.decorator import timer
from common.util.aargparser import get_aargparser


# logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(
    stream=sys.stdout,
    format='[%(asctime)s] %(levelname)s : %(message)s')

#=================================
# 2. Define Class
#=================================

class DailyTemperatureRatioFeature(DataProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_date = self._date_parse('2021-05-01')
        self.table_name = "table_name"
        self.save_type = 'overwrite'
        self.region_mapper = RegionMapper(self.target_date) # custom module

    def filename_for_save(self):
        return f"{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}"

    # TODO: split data type that deliver in dawn 
    @timer
    def import_dlvy_data(self):
        """ aggregation of order for each type of delivery """
        dfs = []
        days = 180
        last_days = days - 1
        for dt in self._date_range(self.start_date, self.end_date, timedelta(days=days)):
            start_date = dt
            end_date = dt + timedelta(days=last_days) if dt + timedelta(days=last_days) < self.target_date else self.target_date
            df = main_read.read_table_with_sql_file('sql_file_name.sql',
                                                     start_date=start_date.to_date,
                                                     end_date=end_date.to_date,
                                                     use_athena=True)
            dfs.append(df)
        dlvy = pd.concat(dfs)
        print('-----complete to import delivery type data-----')

        # handle an exception
        dlvy.dropna(inplace=True) 
        dlvy['column'] = np.where(((dlvy['column'].str.startswith('L')) | (dlvy['column'].str.startswith('N'))), 'CC01', dlvy['column'])
        dlvy['column'] = np.where(((dlvy['column'].str.startswith('L')) | (dlvy['column'].str.startswith('N'))), 'NON_BASIC', dlvy['column'])

        # 센터 맵핑이 안되는 지역 제외
        dlvy = dlvy[(dlvy['column'].str.contains('delivery_type'))]

        dlvy = dlvy.rename(columns={"column": "column"})
        df_dlvy = dlvy.groupby([column_list])['column'].sum().reset_index()
        
        return df_dlvy

    @timer
    def import_storage_data(self):
        """ aggregation for each temperature """
        dfs = []
        days = 180
        last_days = days - 1
        for dt in self._date_range(self.start_date, self.end_date, timedelta(days=days)):
            start_date = dt
            end_date = dt + timedelta(days=last_days) if dt + timedelta(days=last_days) < self.target_date else self.target_date
            df = main_read.read_table_with_sql_file('sql_file_name.sql',
                                                    start_date=start_date.to_date,
                                                    end_date=end_date.to_date,
                                                    use_athena=True)
            dfs.append(df)
        storage = pd.concat(dfs)
        print('-----complete to import storage-----')
        storage = storage[~(storage['column'].str.contains('etc|nodata'))]

        # handle an exception
        storage.dropna(inplace=True)
        storage['column'] = np.where(((storage['column'].str.startswith('L')) | (storage['column'].str.startswith('N'))), 'CC01', storage['column'])
        storage['column'] = np.where(((storage['column'].str.startswith('L')) | (storage['column'].str.startswith('N'))), 'NON_BASIC', storage['column'])
        storage = storage[(storage['column'].str.contains('delivery_type'))]

        # additional preprocessing
        df_coldroom = storage[storage['column'] == 'temperature']
        df_coldroom.pop('column')
        df_coldroom = df_coldroom.rename(columns={"column": "column"})
        df_coldroom = df_coldroom.groupby([columns_list])[[column_list]].sum().reset_index()
        df_coldroom['column'] = df_coldroom['column'] / 2
        storage.pop('column')

        df_storage = storage.groupby([column_list])[[column_list]].sum().reset_index()
        df_storage = pd.concat([df_storage, df_coldroom]).sort_values(
            [column_list]).reset_index(drop=True)
        df_storage = df_storage.rename(columns={"column": "column"})

        return df_storage

    @timer
    def import_promo_data(self):
        """ promotion """
        df_promo = main_read.read_table_with_sql_file('sql_file_name.sql', start_date=self.start_date.to_date)
        print('-----complete to import promotion data-----')
        df_promo = df_promo[[column_list]]
        df_promo = df_promo.drop_duplicates(subset=[column_list], keep='last')
        df_promo['column'] = 1
        df_promo['column'] = pd.to_datetime(df_promo['column'])
        return df_promo

    @timer
    def import_ar_data(self):
        """ marketing kpi """
        ar_goal = read_ar_goal_info_by_ondo_temp(start_date = self.start_date.to_date, end_date = (self.end_date + timedelta(days=30)).to_date, target_date = self.target_date.to_date)
        print('-----complete to import marketing kpi data-----')
        ar_goal['column'] = pd.to_datetime(ar_goal['column'])
        df_ar = ar_goal.sort_values(by=[column_list]).reset_index(drop=True)
        df_ar = df_ar[[column_list]]

        return df_ar

    # TODO : 빅프로모션 진행경과일(ex. 1일차, 2일차 등 ) / 이벤트 이후 경과일 / 프로모션 통합
    @timer
    def import_holiday_data(self):
        """ big promotion and holidays """        
        df_holiday = main_read.read_table_with_sql_file('sql_file_name.sql', start_date=self.start_date.to_date, use_athena = True)\
            .rename(columns={'column': 'column'})
        print('-----complete to import event data-----')

        # remove non-neccessary data 
        df_holiday = df_holiday[(~df_holiday["column"].str.contains('non-neccessary text'))]

        # 프로모션 변수 생성
        df_holiday["column"] = 'test'
        df_holiday.loc[df_holiday["column"].str.contains('text'), 'text'] = 'text'
        df_holiday.loc[df_holiday["column"].str.contains('text'), 'text'] = 'text'
        df_holiday.loc[df_holiday["column"].str.contains('text|text'), 'text'] = 'text'
        df_holiday.loc[df_holiday["column"].str.contains('text'), 'text'] = 'text'
        df_holiday.loc[
            (df_holiday["column"] >= '2022-03-21') & (df_holiday["column"] <= '2022-03-25'), 'text'] = 'flex'
        
        # create holiday variable
        df_holiday.loc[df_holiday["column"].str.contains('text'), "text"] = 'text'
        df_holiday.loc[df_holiday["column"] == "test", "text"] = 'text'
        df_holiday['column'] = pd.to_datetime(df_holiday["column"])
        df_holiday = df_holiday[['column', 'column']]

        # create pre/after holiday variable
        holidays = self._get_before_after_holidays(df_holiday, 'column', 1, 1)
        pub_holidays = self._get_before_after_holidays(df_holiday, 'column', 21, 7)
        pub_holidays = pd.concat([pub_holidays, holidays]).fillna(0)
        df_holiday = df_holiday[~(df_holiday["column"].str.contains('column'))]
        df_holiday = df_holiday.rename(columns={'column': 'column'})
        df_holiday = pd.concat([df_holiday, pub_holidays]).sort_values('column').reset_index(drop=True)
        df_holiday = df_holiday.drop_duplicates(subset='column')

        return df_holiday

    def _get_before_after_holidays(self, df_holiday, value, start, end):

        pub_holiday = df_holiday[df_holiday['column']== value].reset_index(drop=True)
        before = pd.concat([pd.DataFrame({'column': pd.date_range(row.biz_ymd - timedelta(days=start),
                                                    row.biz_ymd - timedelta(days=1), freq='d'),
                           'column': 'column'}, columns=['column', 'column'])
             for i, row in pub_holiday.iterrows()], ignore_index=True)
        after = pd.concat([pd.DataFrame({'column': pd.date_range(row.biz_ymd + timedelta(days=1),
                                                    row.biz_ymd + timedelta(days=end), freq='d'),
                           'column': 'column'}, columns=['column', 'column'])
             for i, row in pub_holiday.iterrows()], ignore_index=True)
        pub_holidays = pd.concat([pub_holiday, before, after], join='outer', axis=0).fillna(0)
        pub_holidays = pub_holidays.drop_duplicates(subset='column').reset_index(drop=True)
        return pub_holidays

    def _pred_dataframe(self, df_ar, df_storage):
        """ expectation period dataframe """
        df_pred = df_ar[df_ar['column'] >= self.target_date]
        df_pred = df_pred.sort_values(by=['column', 'column', 'column'])
        storage = df_storage['column'].unique().tolist()
        df_pred['column'] = [storage] * len(df_pred)
        df_pred = df_pred.explode('column')

        df_pred = df_pred[~((df_pred['column'] == 'text') & (df_pred['column'] == 'text'))]  # BASIC_coldroom 제외
        df_pred = df_pred[['column', 'column', 'column', 'column']]

        return df_pred

    def merge_and_preprocess(self, df_dlvy, df_storage, df_promo, df_ar, df_holiday):
        """ data preprocessing and data join """
        # merge df
        df_total = pd.merge(df_dlvy, df_storage, how="left", on=[column_list]) # merge order 
        df_total['column'] = pd.to_datetime(df_total['column'])
        df_total = df_total[df_total['column'] < self.target_date]
        
        # expectation data merge
        df_pred = self._pred_dataframe(df_ar, df_storage) # test df
        df_total = pd.merge(df_total, df_pred, how='outer', on=[column_list])

        # handle an exception
        df_total = df_total[~((df_total['column'] == 'text') & (df_total['column'] == 'text'))]

        # add variable about date
        df_total['date'] = pd.to_datetime(df_total['biz_ymd'])
        df_total['week_of_month'] = np.ceil(
            (df_total['date'].dt.to_period('M').dt.to_timestamp().dt.weekday + df_total['date'].dt.day) / 7.0)\
            .astype(int)
        df_total['weekday'] = df_total['date'].dt.weekday 
        df_total['month'] = df_total['date'].dt.month 
        df_total = df_total.sort_values(by=[colunm_list])

        # fill nan value with mean value -> add temporary value to future expectation value
        columns = [colunm_list]
        df_groups_month = df_total[df_total['date'] >= self.target_date - timedelta(days=28)].groupby(
            [colunm_list])
        df_groups_3days = df_total[df_total['date'] >= self.target_date - timedelta(days=3)].groupby(
            [colunm_list])
        for c in columns:
            df_total[c] = df_total[c].fillna(df_groups_month[c].transform('median') * 0.65 + df_groups_3days[c].transform('median') * 0.35)
            # In case of holidays, there are some cases that there isn't delivery -> fill with last 1 month's value
            df_total[c] = np.where(df_total[c].isna() == True, df_total[c].fillna(df_groups_month[c].transform('median')), df_total[c])
        
        df_total = df_total.dropna()

        # merge promotion data 
        temp = df_total.merge(df_promo, how="inner", on=[colunm_list])
        temp = temp[[colunm_list]]
        df_total = pd.merge(df_total, temp, how="left", on=[colunm_list])
        df_total['column'] = df_total['column'].fillna(0)

        # merge holiday data
        df_total = pd.merge(df_total, df_holiday, how="left", on=['biz_ymd'])
        df_total.loc[(df_total["biz_ymd"] >= '2022-03-21') & (df_total["biz_ymd"] <= '2022-03-25') & (
            df_total['column'].str.contains('text')), 'text'] = 'save'
 
        df_total[[column_list]] = df_total[[column_list]].fillna('normal')
        df_total = pd.merge(df_total, df_ar, how="left", on=[column_list])
        df_total = df_total[df_total['column'] > 0]

        return df_total

    def preprocess(self,**kwargs):
        """ extract data """
        print(self.target_date.to_date)
        df_dlvy = self.import_dlvy_data()
        df_storage = self.import_storage_data()
        df_promo = self.import_promo_data()
        df_ar = self.import_ar_data()
        df_holiday = self.import_holiday_data()

        """ preprocessing """
        df = self.merge_and_preprocess(df_dlvy, df_storage, df_promo, df_ar, df_holiday) # 조인 및 전처리

        return {'df': df}

    def postprocess(self, **kwargs):
        df = kwargs['df']
        return df

if __name__ == "__main__":
    parser = get_aargparser()
    args = parser.parse_args()
    print('argument: ' + str(args.__dict__))
    # self = DailyTemperatureRatioFeature(target_date = '2023-06-10')
    features = DailyTemperatureRatioFeature(**args.__dict__)
    features.process()
    print(features.result)
