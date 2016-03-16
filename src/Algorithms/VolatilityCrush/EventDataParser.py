import logging
import numpy as np
import pandas as pd

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
    def read_events_from_csv(path: str):
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
    def read_dual_events_from_csv(path_pre: str, path_post: str):

        pass
