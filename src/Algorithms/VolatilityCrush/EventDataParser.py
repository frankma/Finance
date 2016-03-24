import logging

import numpy as np
import pandas as pd

from src.Algorithms.VolatilityCrush.EventDataSABR import EventDataSABR
from src.SABRModel.SABRMarketData import SABRMarketData
from src.Utils.VolType import VolType

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class EventDataParser(object):
    def __init__(self):
        pass

    @staticmethod
    def __date_parser(date_string, date_format=None):
        date_format = '%Y/%m/%d' if date_format is None else date_format
        return pd.datetime.strptime(date_string, date_format)

    @staticmethod
    def load_data(path: str):
        df = pd.read_csv(path, parse_dates=[1, 5], date_parser=EventDataParser.__date_parser)
        df['ID'] = df['ticker'] + '_' + df['asof'].astype(str) + '_' + df['exp'].astype(str)
        df.index = df['ticker']

        columns = list(df.columns)
        ks = []
        for name in columns:
            if 'strike' in name:
                ks.append(name.split('_')[1])
        strikes_name = ['strike' + '_' + v for v in ks]
        vols_name = ['volatility' + '_' + v for v in ks]
        vol_type = VolType.black

        # local method to calibrate model
        def __calibrate_sabr_model(x):
            return SABRMarketData.calibrate_from_vol(x['asof'], x['exp'], x['spot'], x['fwd'], x['bond'],
                                                     np.array(x[strikes_name]).astype(float),
                                                     np.array(x[vols_name]).astype(float), vol_type)

        df['model'] = df.apply(__calibrate_sabr_model, axis=1)

        return df.loc[:, ['asof', 'exp', 'model']].copy()

    @staticmethod
    def load_data_cross_events(path_pre: str, path_post: str):
        df_pre = EventDataParser.load_data(path_pre)
        df_post = EventDataParser.load_data(path_post)
        keys_union = df_pre.index.intersection(df_post.index)
        df = pd.DataFrame(index=keys_union)
        df['EventDataSABR'] = pd.Series(index=keys_union)
        df['Alpha_pre'] = pd.Series(index=keys_union)
        df['Beta_pre'] = pd.Series(index=keys_union)
        df['Nu_pre'] = pd.Series(index=keys_union)
        df['Rho_pre'] = pd.Series(index=keys_union)
        df['Alpha_post'] = pd.Series(index=keys_union)
        df['Beta_post'] = pd.Series(index=keys_union)
        df['Nu_post'] = pd.Series(index=keys_union)
        df['Rho_post'] = pd.Series(index=keys_union)
        for key in keys_union:
            md_pre = [df_pre.loc[key, 'model']]
            md_post = [df_post.loc[key, 'model']]
            df.loc[key, 'EventDataSABR'] = EventDataSABR(md_pre, md_post)
            df.loc[key, 'Alpha_pre'] = df_pre.loc[key, 'model'].alpha
            df.loc[key, 'Beta_pre'] = df_pre.loc[key, 'model'].beta
            df.loc[key, 'Nu_pre'] = df_pre.loc[key, 'model'].nu
            df.loc[key, 'Rho_pre'] = df_pre.loc[key, 'model'].rho
            df.loc[key, 'Alpha_post'] = df_post.loc[key, 'model'].alpha
            df.loc[key, 'Beta_post'] = df_post.loc[key, 'model'].beta
            df.loc[key, 'Nu_post'] = df_post.loc[key, 'model'].nu
            df.loc[key, 'Rho_post'] = df_post.loc[key, 'model'].rho
        return df

    @staticmethod
    def events_stat_analysis(path_pre: str, path_post: str):
        df_pre = EventDataParser.load_data(path_pre)
        df_post = EventDataParser.load_data(path_post)

        keys_union = df_pre.index.intersection(df_post.index)
        time_stamp = ['pre', 'post']
        params = ['md', 'alpha', 'beta', 'nu', 'rho', 'fwd']
        columns_pre = [c + "_" + time_stamp[0] for c in params]
        columns_post = [c + "_" + time_stamp[1] for c in params]
        columns = columns_pre + columns_post
        df = pd.DataFrame(index=keys_union, columns=columns)
        for key in keys_union:
            md_pre = df_pre.loc[key, 'model']
            md_post = df_post.loc[key, 'model']

            df.loc[key, columns_pre] = (md_pre, md_pre.alpha, md_pre.beta, md_pre.nu, md_pre.rho, md_pre.forward)
            df.loc[key, columns_post] = (md_post, md_post.alpha, md_post.beta, md_post.nu, md_post.rho, md_post.forward)

        return df
