import unittest

import matplotlib.pyplot as plt
import numpy as np

from src.Utils.BAW import BAW
from src.Utils.BSM import BSM
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


class MyTestCase(unittest.TestCase):
    def test_error_term(self):
        taus = np.linspace(1.0 / 365.0, 10.0, num=101)
        call_d_h_d_tau_s = np.zeros(np.shape(taus))
        put_d_h_d_tau_s = np.zeros(np.shape(taus))
        s, k, r, q, sig = 100.0, 120.0, 0.08, 0.12, 0.2

        def __h(ttm: float, opt_type: OptionType):
            eta = opt_type.value
            s_opt = BAW.calc_s_optimum(k, ttm, r, q, sig, opt_type)
            lam = BAW.calc_lambda(ttm, r, q, sig, opt_type)
            delta = BSM.delta(s_opt, k, ttm, r, q, sig, opt_type)
            g = 1.0 - np.exp(-r * ttm)
            return (eta - delta) / (g * lam) * s_opt * ((s / s_opt) ** lam)

        dt = 1e-2
        for idx, tau in enumerate(taus):
            t_d = tau * (1.0 - dt)
            t_u = tau * (1.0 + dt)
            call_d_h_d_tau_s[idx] = (__h(t_u, OptionType.call) - __h(t_d, OptionType.call)) / (t_u - t_d)
            put_d_h_d_tau_s[idx] = (__h(t_u, OptionType.put) - __h(t_d, OptionType.put)) / (t_u - t_d)
            self.assertLess(call_d_h_d_tau_s[idx], 3.0)  # loose check for call
            self.assertLess(put_d_h_d_tau_s[idx], 3.0)  # loose check for put

        # plt.plot(taus, call_d_h_d_tau_s, '+-')
        # plt.plot(taus, put_d_h_d_tau_s, 'x-')
        # plt.legend(['call', 'put'])
        plt.show()
        pass

    # recover the cases described in Barone-Adesi and Whaley (1987)
    def test_baw_87_regression(self):
        spots = np.linspace(80, 120, num=5)
        strike = 100

        print('r = 0.08, sig = 0.2, tau = 0.25')
        rate = 0.08
        div = 0.12
        sigma = 0.2
        tau = 0.25
        tgt = dict({80.0: [0.029090206770490012, 0.032151196860539293, 20.413314853565367, 20.418986409015016],
                    90.0: [0.56998560217633631, 0.58964880825250077, 11.249754913486115, 11.251012006296575],
                    100.0: [3.4211088017658966, 3.5249212091037285, 4.3964227775906011, 4.3967493991762883],
                    110.0: [9.8469571518987635, 10.314601714020389, 1.1178157922383782, 1.1179122986820533],
                    120.0: [18.61802275255711, 20.0, 0.18442605741165785, 0.18445776557734184]})
        for spot in spots:
            call_price_ame = BAW.price(spot, strike, tau, rate, div, sigma, OptionType.call)
            call_price_eur = BSM.price(spot, strike, tau, rate, div, sigma, OptionType.call)
            put_price_ame = BAW.price(spot, strike, tau, rate, div, sigma, OptionType.put)
            put_price_eur = BSM.price(spot, strike, tau, rate, div, sigma, OptionType.put)
            print('%r: [%r, %r, %r, %r],'
                  % (spot, call_price_eur, call_price_ame, put_price_eur, put_price_ame))
            benchmarks = tgt.get(spot)
            res = [call_price_eur, call_price_ame, put_price_eur, put_price_ame]
            for idx in range(4):
                self.assertAlmostEqual(benchmarks[idx], res[idx], places=8)

        print('\nr = 0.12, sig = 0.2, tau = 0.25')
        rate = 0.12
        div = 0.16
        sigma = 0.2
        tau = 0.25
        tgt = dict({80.0: [0.028800754376852589, 0.032292609362101737, 20.210198977041813, 20.248165312988096],
                    90.0: [0.56431415067409851, 0.58683842815582399, 11.137817981815843, 11.146198320761224],
                    100.0: [3.3870682004261354, 3.5064279719427738, 4.3526776400446394, 4.3548469766566713],
                    110.0: [9.7489782911725484, 10.28846779323354, 1.1066933392678031, 1.1073321614412424],
                    120.0: [18.432770330907388, 20.0, 0.18259098747943092, 0.18280023647433427]})
        for spot in spots:
            call_price_ame = BAW.price(spot, strike, tau, rate, div, sigma, OptionType.call)
            call_price_eur = BSM.price(spot, strike, tau, rate, div, sigma, OptionType.call)
            put_price_ame = BAW.price(spot, strike, tau, rate, div, sigma, OptionType.put)
            put_price_eur = BSM.price(spot, strike, tau, rate, div, sigma, OptionType.put)
            print('%r: [%r, %r, %r, %r],'
                  % (spot, call_price_eur, call_price_ame, put_price_eur, put_price_ame))
            benchmarks = tgt.get(spot)
            res = [call_price_eur, call_price_ame, put_price_eur, put_price_ame]
            for idx in range(4):
                self.assertAlmostEqual(benchmarks[idx], res[idx], places=8)

        print('\nr = 0.08, sig = 0.40, tau = 0.25')
        rate = 0.08
        div = 0.12
        sigma = 0.4
        tau = 0.25
        tgt = dict({80.0: [1.0461593707874268, 1.0667241939617889, 21.430384017582298, 21.463152714530292],
                    90.0: [3.2321122521104932, 3.2842342654688208, 13.91188156342028, 13.927308928323789],
                    100.0: [7.2910138157937894, 7.4107757079343468, 8.2663277916184938, 8.2741915565381579],
                    110.0: [13.247969543317708, 13.50215573952333, 4.5188281836573303, 4.5231026926655282],
                    120.0: [20.727898128406068, 21.233172650519052, 2.294301433260614, 2.2967515887168464]})
        for spot in spots:
            call_price_ame = BAW.price(spot, strike, tau, rate, div, sigma, OptionType.call)
            call_price_eur = BSM.price(spot, strike, tau, rate, div, sigma, OptionType.call)
            put_price_ame = BAW.price(spot, strike, tau, rate, div, sigma, OptionType.put)
            put_price_eur = BSM.price(spot, strike, tau, rate, div, sigma, OptionType.put)
            print('%r: [%r, %r, %r, %r],'
                  % (spot, call_price_eur, call_price_ame, put_price_eur, put_price_ame))
            benchmarks = tgt.get(spot)
            res = [call_price_eur, call_price_ame, put_price_eur, put_price_ame]
            for idx in range(4):
                self.assertAlmostEqual(benchmarks[idx], res[idx], places=8)

        print('\nr = 0.08, sig = 0.20, tau = 0.50')
        rate = 0.08
        div = 0.12
        sigma = 0.2
        tau = 0.50
        tgt = dict({80.0: [0.20951485896787947, 0.22852362228943432, 20.947296087460288, 20.982170708236463],
                    90.0: [1.3118914043018854, 1.3874002928898812, 12.632027296951819, 12.644527666335504],
                    100.0: [4.4647360111121372, 4.7240711631833259, 6.3672265679195874, 6.3722192281094463],
                    110.0: [10.163461045324198, 10.955248681521992, 2.6483062662891577, 2.6504827866572986],
                    120.0: [17.850543633954942, 20.0, 0.91774351907740659, 0.91876349232285726]})
        for spot in spots:
            call_price_ame = BAW.price(spot, strike, tau, rate, div, sigma, OptionType.call)
            call_price_eur = BSM.price(spot, strike, tau, rate, div, sigma, OptionType.call)
            put_price_ame = BAW.price(spot, strike, tau, rate, div, sigma, OptionType.put)
            put_price_eur = BSM.price(spot, strike, tau, rate, div, sigma, OptionType.put)
            print('%r: [%r, %r, %r, %r],'
                  % (spot, call_price_eur, call_price_ame, put_price_eur, put_price_ame))
            benchmarks = tgt.get(spot)
            res = [call_price_eur, call_price_ame, put_price_eur, put_price_ame]
            for idx in range(4):
                self.assertAlmostEqual(benchmarks[idx], res[idx], places=8)

        print('\nr = 0.08, sig = 0.2, tau = 0.25')
        rate = 0.08
        div = 0.04
        sigma = 0.2
        tau = 0.25
        tgt = dict({80.0: [0.052180027195656065, 0.052180231616756619, 18.868060657937733, 20.0],
                    90.0: [0.8492311361828353, 0.84923216520596778, 9.7646134294332256, 10.183442035906747],
                    100.0: [4.440607561078032, 4.4406119292080879, 3.4554915168367515, 3.5442905881057216],
                    110.0: [11.662215003093678, 11.662231156846968, 0.77660062136072305, 0.79842982303135168],
                    120.0: [20.898288826656852, 20.898342135177387, 0.11217610743220785, 0.11823964549684002]})
        for spot in spots:
            call_price_ame = BAW.price(spot, strike, tau, rate, div, sigma, OptionType.call)
            call_price_eur = BSM.price(spot, strike, tau, rate, div, sigma, OptionType.call)
            put_price_ame = BAW.price(spot, strike, tau, rate, div, sigma, OptionType.put)
            put_price_eur = BSM.price(spot, strike, tau, rate, div, sigma, OptionType.put)
            print('%r: [%r, %r, %r, %r],'
                  % (spot, call_price_eur, call_price_ame, put_price_eur, put_price_ame))
            benchmarks = tgt.get(spot)
            res = [call_price_eur, call_price_ame, put_price_eur, put_price_ame]
            for idx in range(4):
                self.assertAlmostEqual(benchmarks[idx], res[idx], places=8)

        print('\nr = 0.12, sig = 0.2, tau = 0.25')
        rate = 0.12
        div = 0.08
        sigma = 0.2
        tau = 0.25
        tgt = dict({80.0: [0.051660827250086028, 0.051676245047822054, 18.680320317560501, 20.0],
                    90.0: [0.8407811451924303, 0.84085908025505829, 9.6674539024352697, 10.161248537536235],
                    100.0: [4.3964227775906011, 4.3967548428342571, 3.4211088017658966, 3.5254121725416194],
                    110.0: [11.546174024959939, 11.547406184303904, 0.76887331606768683, 0.7944275111188468],
                    120.0: [20.690347378473689, 20.69442613721468, 0.1110599365138869, 0.11813633859889196]})
        for spot in spots:
            call_price_ame = BAW.price(spot, strike, tau, rate, div, sigma, OptionType.call)
            call_price_eur = BSM.price(spot, strike, tau, rate, div, sigma, OptionType.call)
            put_price_ame = BAW.price(spot, strike, tau, rate, div, sigma, OptionType.put)
            put_price_eur = BSM.price(spot, strike, tau, rate, div, sigma, OptionType.put)
            print('%r: [%r, %r, %r, %r],'
                  % (spot, call_price_eur, call_price_ame, put_price_eur, put_price_ame))
            benchmarks = tgt.get(spot)
            res = [call_price_eur, call_price_ame, put_price_eur, put_price_ame]
            for idx in range(4):
                self.assertAlmostEqual(benchmarks[idx], res[idx], places=8)

        print('\nr = 0.08, sig = 0.40, tau = 0.25')
        rate = 0.08
        div = 0.04
        sigma = 0.4
        tau = 0.25
        tgt = dict({80.0: [1.2892877810901027, 1.2894175187550032, 20.105168411832182, 20.52776295599632],
                    90.0: [3.8229825290542863, 3.8232912684473366, 12.738364822304682, 12.926718531910849],
                    100.0: [8.3494057670967763, 8.3500762782444298, 7.3642897228554958, 7.4557092491679722],
                    110.0: [14.796001733931746, 14.797354090727936, 3.910387352198768, 3.9579263502367876],
                    120.0: [22.71360106726263, 22.716167011182133, 1.9274883480379721, 1.953657426662208]})
        for spot in spots:
            call_price_ame = BAW.price(spot, strike, tau, rate, div, sigma, OptionType.call)
            call_price_eur = BSM.price(spot, strike, tau, rate, div, sigma, OptionType.call)
            put_price_ame = BAW.price(spot, strike, tau, rate, div, sigma, OptionType.put)
            put_price_eur = BSM.price(spot, strike, tau, rate, div, sigma, OptionType.put)
            print('%r: [%r, %r, %r, %r],'
                  % (spot, call_price_eur, call_price_ame, put_price_eur, put_price_ame))
            benchmarks = tgt.get(spot)
            res = [call_price_eur, call_price_ame, put_price_eur, put_price_ame]
            for idx in range(4):
                self.assertAlmostEqual(benchmarks[idx], res[idx], places=8)

        print('\nr = 0.08, sig = 0.20, tau = 0.50')
        rate = 0.08
        div = 0.04
        sigma = 0.2
        tau = 0.50
        tgt = dict({80.0: [0.41444622435174239, 0.41447228671909525, 18.077496275043629, 20.0],
                    90.0: [2.1803675541630589, 2.1804484114556697, 10.041430871787405, 10.705752661181382],
                    100.0: [6.4958530768455205, 6.4960756970639721, 4.554929661402312, 4.7720875436372356],
                    110.0: [13.424312207778954, 13.424868692411453, 1.6814020592682084, 1.7603779152009404],
                    120.0: [22.059105411438566, 22.060389801633839, 0.5142085298602499, 0.54557473471989415]})
        for spot in spots:
            call_price_ame = BAW.price(spot, strike, tau, rate, div, sigma, OptionType.call)
            call_price_eur = BSM.price(spot, strike, tau, rate, div, sigma, OptionType.call)
            put_price_ame = BAW.price(spot, strike, tau, rate, div, sigma, OptionType.put)
            put_price_eur = BSM.price(spot, strike, tau, rate, div, sigma, OptionType.put)
            print('%r: [%r, %r, %r, %r],'
                  % (spot, call_price_eur, call_price_ame, put_price_eur, put_price_ame))
            benchmarks = tgt.get(spot)
            res = [call_price_eur, call_price_ame, put_price_eur, put_price_ame]
            for idx in range(4):
                self.assertAlmostEqual(benchmarks[idx], res[idx], places=8)
