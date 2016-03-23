import logging
import sys
from unittest import TestCase

from src.Algorithms.VolatilityCrush.EventDataParser import EventDataParser

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestEventDataParser(TestCase):
    def test_read_events_from_csv(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        path = './Data/sample_2w_pre.csv'
        df = EventDataParser.load_data(path)
        print(df.info)
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_read_dual_events_from_csv(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        path_pre = './Data/sample_2w_pre.csv'
        path_post = './Data/sample_2w_post.csv'
        df = EventDataParser.load_data_cross_events(path_pre, path_post)

        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
