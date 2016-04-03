import logging

from enum import Enum

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class OptionType(Enum):
    call = 1
    put = -1
