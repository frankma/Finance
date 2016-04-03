import logging
from datetime import datetime

import numpy as np

from src.Utils.Types.VolType import VolType

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class IVolatilitySurface(object):
    def __init__(self, asof: datetime):
        self.asof = asof

    def get_vol(self, expiry: datetime, strike: float or np.array, vol_type: VolType):
        pass
