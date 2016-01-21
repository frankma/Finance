import logging
from enum import Enum

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class VolType(Enum):
    black = 'Black Volatility'
    normal = 'Normal Volatility'
