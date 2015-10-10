from pandas import Series, DataFrame

from src.DataObject.DataObjectVolatilitySurfaceEQ import DataObjectVolatilitySurfaceEQ
from src.VolSurface.VolatilitySurface import VolatilitySurface

__author__ = 'frank.ma'


class VolatilitySurfaceEquity(VolatilitySurface):
    def __init__(self, raw_data: DataObjectVolatilitySurfaceEQ):
        super().__init__(raw_data.get_asof())
        self.raw_data = raw_data

        self.expiries = raw_data.expiries
        self.strikes = raw_data.strikes

        self.fwd_b = Series(index=self.expiries)
        self.fwd_a = Series(index=self.expiries)
        self.bond_b = Series(index=self.expiries)
        self.bond_a = Series(index=self.expiries)

        self.call_iv_b = DataFrame(index=self.expiries, columns=self.strikes)
        self.call_iv_a = DataFrame(index=self.expiries, columns=self.strikes)
        self.put_iv_b = DataFrame(index=self.expiries, columns=self.strikes)
        self.put_iv_a = DataFrame(index=self.expiries, columns=self.strikes)

        self.process_data()
        pass

    def process_data(self):
        # firstly process model independent data
        for expiry in self.expiries:
            self.fwd_a[expiry], self.bond_a[expiry] = self.imply_forward_bond(self.raw_data.c_a[expiry],
                                                                              self.raw_data.p_b[expiry])

        # secondly process model dependent data

        pass

    @staticmethod
    def imply_forward_bond(call: Series, put: Series):
        forward, bond = 1, 1
        return forward, bond
