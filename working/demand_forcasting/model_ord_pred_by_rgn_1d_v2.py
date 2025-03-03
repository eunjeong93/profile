"""
Predict the number of orders in next day for each delivery region
"""

import logging
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from datetime import datetime, date, timedelta
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

from common.io.read import main_read
from common.util.aargparser import str2bool
from common.util.aargparser import get_aargparser
from common.batch.dataprocessor import DataProcessor
from common.data.region_mapper import RegionMapper
warnings.filterwarnings("ignore")

class RegionOrderPredict(DataProcessor):


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.start_date = self._date_parse("2022-10-01")        
        self.end_date = self.target_date - timedelta(days=1)
        self.target_min_date = self.target_date - timedelta(days=2)
        self.rgn_mapper = RegionMapper(self.target_date)
        
        self.calibration = '# list of expectation correction values'
        self.calibration =  '# list of expectation correction values'

        self.table_name = "table name"
        self.save_idx = ['column_list']
        self.is_send = kwargs.get("send", True)


    def load_data(self):

        """
        extract exact number of order in each region based on postal code
        """

        df = main_read.read_table_with_sql_file(sql_file_name='sql_file_name.sql',
                                                start_date=self.start_date.to_date, end_date=self.end_date.to_date,
                                                target_min_date=self.target_min_date.to_date)
        df['column'] = pd.to_datetime(df['column'])
        df.rename(columns={'column': 'column'}, inplace=True)
        
        # handle an exception
        df['column'] = np.where((df['column'].str.startswith("L")) | (df['column'].str.startswith("N")), 'text', df['column'])
        df = df[df['column'].isin(self.rgn_list)]  
        df = df.groupby(['column_list'])['column'].sum().reset_index()        

        # additional processing
        df['column'] = df['column'].dt.day_name()
        df['column'] = np.where((df['column'].isin(self.rgn_mapper.NON_BASIC)) & (df['column'] == 'Saturday'), 0, df['column'])

        # remove today data: add to_date 
        df = df.query(
            "biz_ymd < @self.end_date.to_date").drop(['column'], axis=1)

        # correct data according to inner rule
        try:
            conti = main_read.read_table_with_sql_file(sql_file_name='sql_file_name.sql',
                                                       start_date=self.start_date.to_date, end_date=self.end_date.to_date)
            conti['column'] = pd.to_datetime(conti['column'])

            # 센터의 권역들을 받아와 처리
            # 컨티값 보정: 수도권샛별만 필요 (부울도 보정 들어감)
            # handle exception
            
        except:
            pass

        logging.info(" finished")
        return df
    
    

    def load_ar_info(self):
        
        """
        import marketing data
        """
        # TODO: 34center
        sql = f"""
            select
                column
            from table
            where biz_ymd::date > '{self.start_date.to_date}'
            order by biz_ymd
            """
        ar_goal = main_read.read_sql_query(sql)
        ar_goal = ar_goal.query("row_number == 1")
        ar_goal['column'] = pd.to_datetime(ar_goal['biz_ymd'])
        ar_goal['column'] = ar_goal['column'].astype(int)
        ar = ar_goal.drop(['column'], axis=1)
        logging.info(" finished")
        return ar
    
    

    def make_future_dataset(self, df, max_date):

        future_date = pd.Series(pd.date_range(self.end_date.to_date, max_date))
        biz_ymd_list = df['column'].drop_duplicates().append(future_date).drop_duplicates().to_frame().reset_index(drop=True)
        biz_ymd_list.columns = ['column']

        biz_hour_list = df['column'].drop_duplicates().to_frame()
        biz_rgn_list = df['column'].drop_duplicates().to_frame()

        base_df = pd.merge(biz_ymd_list, biz_hour_list, how='cross').merge(biz_rgn_list, how='cross')
        df = base_df.merge(df, on=['column', 'column', 'column'], how='left')\
                    .sort_values(['column', 'column', 'column'])

        # past date : 0, future date : nan
        df.loc[df.biz_ymd < self.end_date.to_date] = df.loc[df.biz_ymd < self.end_date.to_date].fillna(0)
        logging.info(" finished")

        return df



    def add_fe_ar_goal(self, df, ar):

        """
        Use marketing data as a feature
        """
        df = df.merge(ar, on='biz_ymd', how='left')

        df['column'] = df['column'] / df['column']
        df['column'] = df.groupby(['column', 'column'])['column'].shift(7)
        df['column'] = df.groupby(['column', 'column'])['column'].shift(14)
        df['column'] = df.groupby(['column', 'column'])['column'].shift(21)
        df['column'] = df[['column', 'column']].mean(axis=1)
        df['column'] = df[['column','column', 'column']].mean(axis=1)

        df['column'] = df['column'] * df['column']
        df['column'] = df['column'] * df['column']
        logging.info(" finished")

        return df



    def add_fe_daterelated(self, df):

        """
        add date feature
        """
        df['biz_ymd'] = pd.to_datetime(df['biz_ymd'])
        df["month"] = df['biz_ymd'].dt.month #월
        df["week_no"] = df['biz_ymd'].dt.isocalendar()['week'].astype('int')
        df['week_num'] = np.ceil((df['biz_ymd'].dt.to_period('M').dt.to_timestamp().dt.weekday + df['biz_ymd'].dt.day) / 7.0).astype(int)  # 월별주차
        df["dayofweek"] = df['biz_ymd'].dt.day_name() # 요일

        df = pd.concat([df, pd.get_dummies(data=df['dayofweek'], prefix='dayofweek')], axis=1)  # 요일 one-hot encoding
        df["weekend_yn"] = np.where(df["dayofweek"].isin(["Saturday", "Sunday"]), 1, 0)  # 주말여부
        df["dayofyear"] = df['biz_ymd'].dt.dayofyear  # 연간 일
        df['dayofweek'] = LabelEncoder().fit_transform(df['dayofweek'])

        logging.info(" finished")
        return df



    def add_fe_holiday(self, df):

        """
        add holiday, promotion etc
        """
        holiday_df = main_read.read_table_with_sql_file(sql_file_name='sql_file_name.sql',
                                                        start_date=self.start_date.to_date,
                                                        use_athena=True).rename(columns={'column': 'column'})

        holiday_df['column'] = pd.to_datetime(holiday_df['column'])

        # distingiush event and holiday
        holiday_df_kurly = holiday_df[holiday_df.holiday.str.contains('text')].groupby(['column'])['column'].apply(lambda x: '&'.join(x)).reset_index()
        holiday_df_public = holiday_df[~holiday_df.holiday.str.contains('text', na=True)].groupby(['column'])['column'].apply(lambda x: '&'.join(x)).reset_index()

        # holiday variable
        df = df.merge(holiday_df_public, 'left', on='column')
        df['column'] = 0
        df['column'] = 0
        df.loc[df.holiday.str.contains('text', na=False), 'column'] = 1
        df.loc[~df.holiday.str.contains('text', na=True), 'column'] = 1
        df.drop(columns='holiday', inplace=True)

        # event variable
        df = df.merge(holiday_df_kurly, 'left', on='column')
        df['column'] = 0
        df['column'] = 0
        df['column'] = 0
        df['column'] = 0
        df.loc[df.holiday.str.contains('text', na=False), 'text'] = 1
        df.loc[df.holiday.str.contains('text', na=False), 'text'] = 1
        df.loc[df.holiday.str.contains('text', na=False), 'text'] = 1
        df.loc[df.holiday.str.contains('text', na=False), 'text'] = 1
        df.drop(columns='column', inplace=True)

        # traditional holiday previous, after 7days
        df['column'] = np.nan
        df['column'] = np.nan
        for day in df[df.traditional_holiday_yn == 1].biz_ymd.unique():
            day = pd.to_datetime(day)
            bf14 = (day - pd.Timedelta(days=14)).strftime(self.date_format)
            bf1 = (day - pd.Timedelta(days=1)).strftime(self.date_format)
            af7 = (day + pd.Timedelta(days=7)).strftime(self.date_format)
            af1 = (day + pd.Timedelta(days=1)).strftime(self.date_format)
            df.loc[(df.biz_ymd >= bf14) & (df.biz_ymd <= bf1) & (df.traditional_holiday_yn == 0), 'before_holiday'] = 1
            df.loc[(df.biz_ymd >= af1) & (df.biz_ymd <= af7) & (df.traditional_holiday_yn == 0), 'after_holiday'] = 1

        df[['column', 'column']] = df[['column', 'column']].fillna(0)
        logging.info("finished")
        return df



    def add_fe_laggging(self, df, max_lagging_n=30):

        """
        add lagging feature and rolling mean feature
        """
        for n in np.arange(2, max_lagging_n):

            # lagging values
            df[f'ord_cnt_d{n}'] = df.groupby(['column', 'column'])['column'].shift(n)
            df[f'ord_cnt_roll_d{n}'] = df.groupby(['column', 'column'])['column'].transform(lambda x: x.rolling(n, 1).mean())
            df[f'ord_cnt_ewm_d{n}'] = df.groupby(['column', 'column'])['column'].transform(lambda x: x.ewm(span=n, adjust=False).mean())

        logging.info("finished")
        return df



    def add_fe_change_ratio(self, df, max_change_n=14):

        """
        add feature indicating the rate of increase/decrease
        """
        for n in np.arange(2, max_change_n):

            df[f'ord_cnt_change_d1{n}'] = (df[f'ord_cnt_d{n}']-df[f'ord_cnt_d{n+1}'])/df[f'ord_cnt_d{n+1}']
            df[f'ord_cnt_change_d2{n}'] = (df[f'ord_cnt_d{n+1}']-df[f'ord_cnt_d{n+2}'])/df[f'ord_cnt_d{n+2}']
            df[f'ord_cnt_change_d3{n}'] = (df[f'ord_cnt_d{n+2}']-df[f'ord_cnt_d{n+3}'])/df[f'ord_cnt_d{n+3}']
            df[f'ord_cnt_change_d4{n}'] = (df[f'ord_cnt_d{n+3}']-df[f'ord_cnt_d{n+4}'])/df[f'ord_cnt_d{n+4}']

        df[f'ord_cnt_change_w1'] = (df[f'ord_cnt_d2']-df[f'ord_cnt_d7'])/df[f'ord_cnt_d7']
        df[f'ord_cnt_change_w2'] = (df[f'ord_cnt_d7']-df[f'ord_cnt_d14'])/df[f'ord_cnt_d14']
        df[f'ord_cnt_change_w3'] = (df[f'ord_cnt_d14']-df[f'ord_cnt_d21'])/df[f'ord_cnt_d21']

        logging.info("finished")
        return df



    def add_fe_rgn_ratio(self, df):

        """
        consider the rete of order in each region
        """


        n_list = [2, 3, 5, 7, 10, 11, 14, 21]
        for n in n_list:

            df[f'ord_sum_d{n}'] = df.groupby(['region_group_code', 'biz_hour'])['ord_sum'].shift(n)
            df[f'ord_ratio_d{n}'] = df.groupby(['region_group_code', 'biz_hour'])['ord_ratio'].shift(n)
            df[f'ord_sum_roll_d{n}'] = df.groupby(['region_group_code', 'biz_hour'])['ord_sum_d2'].transform(lambda x: x.rolling(n,1).mean())
            df[f'ord_ratio_roll_d{n}'] = df.groupby(['region_group_code', 'biz_hour'])['ord_ratio_d2'].transform(lambda x: x.rolling(n,1).mean())

        logging.info("finished")
        return df



    def add_fe_promotion(self, df):

        """
        add feature about promotion information
        """

        promo = main_read.read_table_with_sql_file(sql_file_name='sql_file_name.sql', start_date=self.start_date.to_date)
        promo = promo.drop_duplicates()

        # 프로모션 별 시작, 끝 날짜밖에 없으므로 별도로 df 생성
        promo_df = pd.DataFrame()
        for i in promo.index:
            subset = promo.loc[i]
            start = str(subset.loc['column'])
            end = str(subset.loc['column'])

            sub_df = pd.Series(pd.date_range(start, end)).to_frame()
            sub_df.columns = ['column']
            sub_df['column'] = subset.loc['column']
            sub_df['column'] = subset.loc['column']

            promo_df = promo_df.append(sub_df)

        promo_df = promo_df.groupby(['column_list'])['column'].nunique().unstack().reset_index().fillna(0)
        promo_df.rename(columns={'column': 'column'}, inplace=True)

        df = df.merge(promo_df, on=['column'], how='left')

        logging.info("finished")
        return df



    def add_fe_mkt_event(self, df):

        """
        add additional marketing variable
        """

        sql = f"""
                select
                    column
                from table
                where ymd::date >= '{self.start_date.to_date}'
                  and update_dt=(select max(update_dt) from table)
                """

        mkt = main_read.read_sql_query(sql=sql)
        mkt['column'] = pd.to_datetime(mkt['column'])
        mkt = mkt.sort_values("column")

        mkt['column'] = np.where(mkt['column'].str.contains("text"), 1, 0)
        mkt['column'] = np.where(mkt['column'].str.contains("text"), 1, 0)
        mkt['column'] = np.where(mkt['column'] != '', 1, 0)
        df = df.merge(mkt, on='column', how='left')

        logging.info("finished")
        return df


    def preprocess(self, **kwargs):

        """
        after generating featureset, train the data and expect value
        """
        # featureset 생성
        df = self.load_data()
        ar = self.load_ar_info()
        df = self.make_future_dataset(df, ar.biz_ymd.max().strftime(self.date_format))
        df = self.add_fe_ar_goal(df, ar)
        df = self.add_fe_daterelated(df)
        df = self.add_fe_holiday(df)
        df = self.add_fe_laggging(df)
        df = self.add_fe_change_ratio(df)
        df = self.add_fe_rgn_ratio(df)
        df = self.add_fe_promotion(df)
        df = self.add_fe_mkt_event(df)
        df = df.set_index("biz_ymd")

        # after filtering in each region, train/inferrence
        pred_res = pd.DataFrame()
        rgn_list = df.region_group_code.drop_duplicates()
        target_encoder = ce.TargetEncoder(cols=['column'])
        drop_cols = ['column', 'column', 'column', 'column', 'column']

        # enter the specific date
        for date in pd.period_range(self.target_min_date.to_date, self.target_date.to_date):
            logging.info(f"{date}: model train & predict is started")
            for rgn in rgn_list:

                # subset
                subset = df.query("region_group_code == @rgn")

                # exclude today date
                
                train = subset[subset.index < str(date-1)] # previous date
                pred = subset[subset.index >= str(date)] # target date

                X_train = train.drop(drop_cols, axis=1)
                X_train['column'] = target_encoder.fit_transform(train['column'], train['column'])
                y_train = train[['column']]

                X_pred = pred.drop(drop_cols, axis=1)
                X_pred['column'] = target_encoder.transform(pred['column'])
                y_pred = pred[['column', 'column', 'column']]

                #todo: optimize parameter
                model = LGBMRegressor(boosting_type='gbdt',
                                      random_state=42)
                                      #n_estimators = 1500,
                                      #learning_rate=0.05,
                model.fit(X_train, y_train)

                y_pred['pred'] = model.predict(X_pred).astype(int)
                y_pred['pred_date'] = str(date-1) # 예측수행일
                pred_res = pred_res.append(y_pred)

            logging.info(f"{date}: model train & predict is finished")
        pred_res = pred_res.reset_index()

        pred_res['biz_ymd'] = pd.to_datetime(pred_res['biz_ymd'])
        pred_res['pred_date'] = pd.to_datetime(pred_res['pred_date'])
        
        return {'df': pred_res, 'ar': ar}



    def postprocess(self, **kwargs):

        """
        correct specific value and deliver the expectation value to Slack
        """
        df = kwargs['df']
        ar = kwargs['ar']

        # additional handling
        cc_regions = self.rgn_mapper.get_center_code_per_region_group_code()
        cc_dlvy_type = self.rgn_mapper.get_dlvy_type_per_region_group_code()
        df['column'] = df['column']
        df['column'] = df['column']
        df.replace({'center_cd' : cc_regions, 'dlvy_type': cc_dlvy_type}, inplace=True)

        # remove negative value
        df['column'] = np.where(df['column'] < 0, np.nan, df['pred'])
        df['pred'] = df['pred'].interpolate()

        # holiday value makes 0
        df['dayofweek'] = df['column'].dt.day_name()
        df['pred'] = np.where((df['column'] == 'text') & (df['column'] == 'text'), 0, df['pred'])

        # 센터별 보정치 적용: 1차
        # remove code
        
        
        if self.is_send:
            
            ## provide expectation value to Slack ##            
            df_send = df.query("biz_ymd == @self.target_date.to_date").query("pred_date == @self.end_date.to_date")
            # remove code                 

        
            #TODO: add new region
            # order the region with number
            order_list = {}

            df_send['column'] = df_send['column'].replace(order_list)
            df_send = df_send.sort_values("column") 
            df_send.index = df_send['column']  

            payload = {
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"""
                            \n Hello? Tomorrow ({self.target_date.to_date})expected value is.\
                            \n These values are expected value from data science team below.\
                            \n [retion1] expected value\
                            \n  • A: {df_send['pred_rev'][0].astype(int)}\
                            \n  • C: {df_send['pred_rev'][1].astype(int)}\
                            \n  • X: {df_send['pred_rev'][2].astype(int)}\
                            \n  • M: {df_send['pred_rev'][3].astype(int)}\
                            \n  • I: {df_send['pred_rev'][4].astype(int)}\
                            \n  • Y: {df_send['pred_rev'][5].astype(int)}\
                            \n [retion2] expected value\
                            \n  • R: {df_send['pred_rev'][11].astype(int)}\
                            \n  • F: {df_send['pred_rev'][12].astype(int)}\
                            \n  • W: {df_send['pred_rev'][13].astype(int)}\
                            \n  • H: {df_send['pred_rev'][14].astype(int)}\
                            \n  • D: {df_send['pred_rev'][15].astype(int)}\
                            \n  • B: {df_send['pred_rev'][16].astype(int)}\
                            \n  • S: {df_send['pred_rev'][17].astype(int)}\
                            \n  • T: {df_send['pred_rev'][18].astype(int)}\
                            \n  • G: {df_send['pred_rev'][19].astype(int)}\
                            \n  • Z: {df_send['pred_rev'][20].astype(int)}\
                            \n  • TK: {df_send['pred_rev'][22].astype(int)}\
                            \n [retion3] expected value\
                            \n  • DHL: {df_send['pred_rev'][6].astype(int)}\
                            \n  • AA: {df_send['pred_rev'][7].astype(int)}\
                            \n  • DWDE: {df_send['pred_rev'][8].astype(int)}\
                            \n  • BS: {df_send['pred_rev'][9].astype(int)}\
                            \n  • UL: {df_send['pred_rev'][10].astype(int)}\
                            \n  • GC: {df_send['pred_rev'][21].astype(int)}\
                            \n [Total] {df_send['pred_rev'].sum().astype(int)}\
                            \n ---------------------------------------------\
                                """,
                        }
                    }],
                "attachments": [
                    {
                        "color": "#7649AD",
                        "blocks": [
                            {
                                "type": "context",
                                "elements": [
                                    {
                                        "type": "plain_text",
                                        "text": f"● text"
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }

            channel_name1 = "channel_name" if not self.test else "test_channel"
            self.send_slack_message(bot_name='bot_name', channel=channel_name1, message=payload)

        df['model_date'] = self.end_date.to_date
        return df
        

if __name__ == "__main__":
    parser = get_aargparser(target_date=datetime.now() + timedelta(days=1))
    parser.add_argument('--send', type=str2bool, default=True)
    args = parser.parse_args()

    cls = RegionOrderPredict(**args.__dict__)
    cls.process()