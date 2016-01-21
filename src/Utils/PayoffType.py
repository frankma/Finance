import logging

from enum import Enum

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class PayoffType(Enum):
    European = 'European'
    American = 'American'
    Binary = 'Binary'
    CashOrNothing = 'Cash-or-Nothing'
    AssetOrNothing = 'Asset-or-Nothing'
