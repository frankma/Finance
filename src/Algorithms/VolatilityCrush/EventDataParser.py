import numpy as np
import pandas as pd

from src.SABRModel.SABRMarketData import SABRMarketData
from src.Utils.VolType import VolType

__author__ = 'frank.ma'


class EventDataParser(object):
    def __init__(self):
        pass

    @staticmethod
    def __date_parser(date_string, date_format=None):
        date_format = '%Y/%m/%d' if date_format is None else date_format
        return pd.datetime.strptime(date_string, date_format)

    @staticmethod
    def read_events_from_csv(path: str):
        df = pd.read_csv(path, parse_dates=[1, 5], date_parser=EventDataParser.__date_parser)
        df['ID'] = df['ticker'] + '_' + df['asof'].astype(str) + '_' + df['exp'].astype(str)
        df.index = df['ticker'] + '_' + df['exp'].astype(str)

        columns = list(df.columns)
        ks = []
        for name in columns:
            if 'strike' in name:
                ks.append(name.split('_')[1])
        strikes_name = ['strike' + '_' + v for v in ks]
        vols_name = ['volatility' + '_' + v for v in ks]

        df_out = pd.DataFrame(index=df.index)
        df_out['model'] = pd.Series(index=df.index)
        for idx, row in df.iterrows():
            asof, expiry = row['asof'], row['exp']
            spot, forward, bond = row['spot'], row['fwd'], row['bond']
            strikes = np.array(row[strikes_name]).astype(float)
            volatilities = np.array(row[vols_name]).astype(float)
            vol_type = VolType.black
            df_out['model'].loc[idx] = SABRMarketData.calibrate_from_vol(asof, expiry, spot, forward, bond, strikes,
                                                                         volatilities, vol_type, beta=1.0)
        return df_out

    @staticmethod
    def read_dual_events_from_csv(path_pre: str, path_post: str):

        pass
