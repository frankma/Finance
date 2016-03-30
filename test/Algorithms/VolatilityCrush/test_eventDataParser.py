import logging
import sys
from unittest import TestCase

import numpy as np
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
        data_src = ''  # 'sample_'
        horizons = ['1m']  # ['2w', '1m']
        for horizon in horizons:
            logger.info('processing expiry of %s' % horizon)
            path_pre = './Data/%s%s_pre.csv' % (data_src, horizon)
            path_post = './Data/%s%s_post.csv' % (data_src, horizon)
            df = EventDataParser.events_stat_analysis(path_pre, path_post, filtration='yhoo')
            plt.figure(horizon)
            for idx, name in enumerate(['alpha', 'nu', 'rho']):
                ax = plt.subplot(2, 2, idx + 1)
                v_pre = df[name + '_pre']
                v_post = df[name + '_post']
                v_max = max(max(v_pre), max(v_post))
                v_min = min(min(v_pre), min(v_post))
                plt.scatter(v_pre, v_post, marker='o', alpha=1.0)
                for ticker in df.index:
                    ax.annotate(ticker, (v_pre[ticker], v_post[ticker]), alpha=0.4, color='k')
                plt.plot([v_min, v_max], [v_min, v_max], c='k', alpha=0.5)
                plt.xlim([v_min, v_max])
                plt.ylim([v_min, v_max])
                plt.xlabel(name + '_pre')
                plt.ylabel(name + '_post')
                plt.title(name)
            # forward
            ax = plt.subplot(2, 2, 4)
            fwd_pre = df['fwd_pre']
            fwd_post = df['fwd_post']
            ret = fwd_post / fwd_pre
            plt.plot(list(ret), 'o', alpha=1.0)
            for ndx, ticker in enumerate(df.index):
                ax.annotate(ticker, (ndx, ret[ndx]), alpha=0.4, color='k')
            plt.plot([0, df.__len__()], [1.0, 1.0], c='k', alpha=0.5)
            plt.title('fwd return')

            rel_strikes = np.linspace(0.6, 1.4, num=21)
            f, axes = plt.subplots(4, 4, sharey=True)
            f.canvas.set_window_title('%s implied vol' % horizon)
            for idx, key in enumerate(df.index):
                model_pre = df.loc[key, 'md_pre'].model
                fwd_pre = df.loc[key, 'fwd_pre']
                strikes_pre = fwd_pre * rel_strikes
                vols_pre = model_pre.calc_vol_vec(fwd_pre, strikes_pre)

                model_post = df.loc[key, 'md_post'].model
                fwd_post = df.loc[key, 'fwd_post']
                strikes_post = fwd_post * rel_strikes
                vol_post = model_post.calc_vol_vec(fwd_post, strikes_post)

                atm_vol_pre = model_pre.calc_vol(fwd_pre, fwd_pre)
                atm_vol_shock = model_post.calc_vol(fwd_post, fwd_pre)
                atm_vol_post = model_post.calc_vol(fwd_post, fwd_post)

                i = int(idx / 4)
                j = idx % 4
                axes[i][j].set_title(key)
                axes[i][j].plot(strikes_pre, vols_pre, '-', color='b', label='pre')
                axes[i][j].plot([fwd_pre, fwd_pre], [atm_vol_pre, atm_vol_shock], 'o--', color='b', label='pre_atm')
                axes[i][j].plot(strikes_post, vol_post, '-', color='r', label='post')
                axes[i][j].plot(fwd_post, atm_vol_post, 'o', color='r', label='post_atm')
                axes[i][j].legend(fontsize='x-small')

        plt.show()
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
