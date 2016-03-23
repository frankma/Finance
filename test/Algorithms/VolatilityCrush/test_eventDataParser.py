import logging
import sys
from unittest import TestCase
import matplotlib.pyplot as plt

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
        alpha_pre = df['Alpha_pre']
        alpha_post = df['Alpha_post']
        plt.scatter(alpha_pre, alpha_post, label='alpha', c='r')
        nu_pre = df['Nu_pre']
        nu_post = df['Nu_post']
        plt.scatter(nu_pre, nu_post, label='nu', c='b')
        rho_pre = df['Rho_pre']
        rho_post = df['Rho_post']
        plt.scatter(rho_pre, rho_post, label='rho', c='g')
        plt.plot(range(-2, 5), range(-2, 5), c='k')
        plt.legend()
        plt.show()
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
