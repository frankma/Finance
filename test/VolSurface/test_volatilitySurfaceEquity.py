from unittest import TestCase
from datetime import datetime

from pandas import Series, DataFrame

from src.DataObject.DataObjectVolatilitySurfaceEQ import DataObjectVolatilitySurfaceEQ
from src.VolSurface.VolatilitySurfaceEquity import VolatilitySurfaceEquity

__author__ = 'frank.ma'


class TestVolatilitySurfaceEquity(TestCase):
    def test_process_data(self):
        # TODO: find a real set of data to feed in

        asof = datetime(2015, 2, 16)

        expiries = [datetime(2015, 3, 27), datetime(2015, 3, 28)]
        strikes = [2.20, 2.25, 2.30]

        call_bid = DataFrame([[0.21, 0.19, 0.17], [0.31, 0.28, 0.25]], index=expiries, columns=strikes)
        call_ask = DataFrame([[0.21, 0.19, 0.17], [0.31, 0.28, 0.25]], index=expiries, columns=strikes)
        call_volume = DataFrame([[10, 25, 32], [43, 32, 12]], index=expiries, columns=strikes)
        put_bid = DataFrame([[0.21, 0.19, 0.17], [0.31, 0.28, 0.25]], index=expiries, columns=strikes)
        put_ask = DataFrame([[0.21, 0.19, 0.17], [0.31, 0.28, 0.25]], index=expiries, columns=strikes)
        put_volume = DataFrame([[67, 12, 23], [12, 56, 25]], index=expiries, columns=strikes)

        data = DataObjectVolatilitySurfaceEQ(asof, Series(expiries, index=expiries), Series(strikes, index=strikes),
                                             call_bid, call_ask, call_volume, put_bid, put_ask, put_volume)

        vs_eq = VolatilitySurfaceEquity(data)
        vs_eq.process_data()
