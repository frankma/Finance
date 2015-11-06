from enum import Enum

__author__ = 'frank.ma'


class PayoffType(Enum):
    European = 'European'
    American = 'American'
    Binary = 'Binary'
    CashOrNothing = 'Cash-or-Nothing'
    AssetOrNothing = 'Asset-or-Nothing'
