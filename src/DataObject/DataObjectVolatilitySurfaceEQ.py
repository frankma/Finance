import datetime
from src.DataObject.DataObject import DataObject
from pandas import Series, DataFrame

__author__ = 'frank.ma'


class DataObjectVolatilitySurfaceEQ(DataObject):

    def __init__(self, asof: datetime, expiries: Series, strikes: Series,
                 call_bid: DataFrame, call_ask: DataFrame, call_volume: DataFrame,
                 put_bid: DataFrame, put_ask: DataFrame, put_volume: DataFrame):
        super().__init__(asof)

        self.expiries = expiries
        self.strikes = strikes

        self.c_b = call_bid
        self.c_a = call_ask
        self.c_vlm = call_volume
        self.p_b = put_bid
        self.p_a = put_ask
        self.p_vlm = put_volume