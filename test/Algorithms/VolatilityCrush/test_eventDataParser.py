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
        print(df.info())
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_event_stat_analysis(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        path_pre = './Data/sample_2w_pre.csv'
        path_post = './Data/sample_2w_post.csv'
        df = EventDataParser.events_stat_analysis(path_pre, path_post)
        print(df.info())
        for idx, name in enumerate(['alpha', 'nu', 'rho']):
            ax = plt.subplot(2, 2, idx + 1)
            v_pre = df[name + '_pre']
            v_post = df[name + '_post']
            v_max = max(max(v_pre), max(v_post))
            v_min = min(min(v_pre), min(v_post))
            # plt.plot(list(v_pre), 'x', c='b', label='pre')
            # plt.plot(list(v_post), '+', c='r', label='post')
            # plt.ylim([v_min - 0.1, v_max + 0.1])
            # plt.legend()
            plt.scatter(v_pre, v_post, marker='x', alpha=1.0)
            for ticker in df.index:
                ax.annotate(ticker, (v_pre[ticker], v_post[ticker]), alpha=0.2, color='red')
            plt.plot([v_min, v_max], [v_min, v_max], c='k', alpha=0.5)
            plt.xlim([v_min - 0.1, v_max + 0.1])
            plt.ylim([v_min - 0.1, v_max + 0.1])
            plt.xlabel(name + '_pre')
            plt.ylabel(name + '_post')
            plt.title(name)
        # forward
        ax = plt.subplot(2, 2, 4)
        fwd_pre = df['fwd_pre']
        fwd_post = df['fwd_post']
        ret = fwd_post / fwd_pre
        plt.plot(list(ret), '.', alpha=1.0)
        for ndx, ticker in enumerate(df.index):
            ax.annotate(ticker, (ndx, ret[ndx]), alpha=0.2, color='red')
        plt.plot([0, df.__len__()], [1.0, 1.0], c='k', alpha=0.5)
        plt.title('fwd return')

        plt.show()
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
