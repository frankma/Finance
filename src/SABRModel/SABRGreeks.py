import numpy as np

from src.SABRModel.SABRModel import SABRModelLognormalApprox
from src.Utils.Types.OptionType import OptionType
from src.Utils.Types.VolType import VolType
from src.Utils.Valuator.Black76 import Black76Vec

__author__ = 'frank.ma'


class SABRGreeks(object):
    @staticmethod
    def __check_model(model: SABRModelLognormalApprox):
        beta = model.beta
        if beta != 1:
            raise NotImplementedError('sabr model parameter beta (%r) should be one.' % beta)
        pass

    @staticmethod
    def theta(f: float, k: np.array, tau: float, b: float, opt_type: OptionType,
              model: SABRModelLognormalApprox) -> np.array:
        SABRGreeks.__check_model(model)
        sig = model.calc_vol_vec(f, k, vol_type=VolType.black)
        theta = Black76Vec.theta(f, k, tau, sig, b, opt_type)
        vega = Black76Vec.vega(f, k, tau, sig, b)
        d_black_d_t = model.calc_d_black_d_t(f, k)
        return theta + vega * d_black_d_t

    @staticmethod
    def delta(f: float, k: np.array, tau: float, b: float, opt_type: OptionType,
              model: SABRModelLognormalApprox) -> np.array:
        SABRGreeks.__check_model(model)
        sig = model.calc_vol(f, k, vol_type=VolType.black)
        delta = Black76Vec.delta(f, k, tau, sig, b, opt_type)
        vega = Black76Vec.vega(f, k, tau, sig, b)
        d_black_d_f = model.calc_d_black_d_k(f, k)
        return delta + vega * d_black_d_f

    @staticmethod
    def delta_k(f: float, k: np.array, tau: float, b: float, opt_type: OptionType,
                model: SABRModelLognormalApprox) -> np.array:
        SABRGreeks.__check_model(model)
        sig = model.calc_vol_vec(f, k, vol_type=VolType.black)
        delta_k = Black76Vec.delta_k(f, k, tau, sig, b, opt_type)
        vega = Black76Vec.vega(f, k, tau, sig, b)
        d_black_d_k = model.calc_d_black_d_k(f, k)
        return delta_k + vega * d_black_d_k

    @staticmethod
    def gamma(f: float, k: np.array, tau: float, b: float, model: SABRModelLognormalApprox) -> np.array:
        SABRGreeks.__check_model(model)
        sig = model.calc_vol_vec(f, k, vol_type=VolType.black)
        gamma = Black76Vec.gamma(f, k, tau, sig, b)
        vanna = Black76Vec.vanna(f, k, tau, sig, b)
        vomma = Black76Vec.vomma(f, k, tau, sig, b)
        vega = Black76Vec.vega(f, k, tau, sig, b)
        d_black_d_f = model.calc_d_black_d_f(f, k)
        d2_black_d_f2 = model.calc_d2_black_d_f2(f, k)
        return gamma + 2.0 * vanna * d_black_d_f + vomma * (d_black_d_f ** 2) + vega * d2_black_d_f2

    @staticmethod
    def gamma_k(f: float, k: np.array, tau: float, b: float, model: SABRModelLognormalApprox) -> np.array:
        SABRGreeks.__check_model(model)
        sig = model.calc_vol_vec(f, k, vol_type=VolType.black)
        gamma_k = Black76Vec.gamma_k(f, k, tau, sig, b)
        vanna_k = Black76Vec.vanna_k(f, k, tau, sig, b)
        vomma = Black76Vec.vomma(f, k, tau, sig, b)
        vega = Black76Vec.vega(f, k, tau, sig, b)
        d_black_d_k = model.calc_d_black_d_k(f, k)
        d2_black_d_k2 = model.calc_d2_black_d_k2(f, k)
        return gamma_k + 2.0 * vanna_k * d_black_d_k + vomma * (d_black_d_k ** 2) + vega * d2_black_d_k2
