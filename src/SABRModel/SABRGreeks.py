import numpy as np

from src.SABRModel.SABRModel import SABRModelLognormalApprox
from src.Utils.OptionType import OptionType
from src.Utils.Valuator.Black76 import Black76Vec
from src.Utils.VolType import VolType

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
    def delta_k(f: float, k: np.array, tau: float, b: float, opt_type: OptionType,
                model: SABRModelLognormalApprox) -> np.array:
        SABRGreeks.__check_model(model)
        sig = model.calc_vol_vec(f, k, vol_type=VolType.black)
        delta_k = Black76Vec.delta_k(f, k, tau, sig, b, opt_type)
        vega = Black76Vec.vega(f, k, tau, sig, b)
        d_black_d_k = model.calc_d_black_d_k(f, k)
        return delta_k + vega * d_black_d_k

    @staticmethod
    def gamma_k(f: float, k: np.array, tau: float, b: float, model: SABRModelLognormalApprox) -> np.array:
        SABRGreeks.__check_model(model)
        sig = model.calc_vol_vec(f, k, vol_type=VolType.black)
        gamma_k = Black76Vec.gamma_k(f, k, tau, sig, b)
        vanna = Black76Vec.vanna(f, k, tau, sig, b)
        vomma = Black76Vec.vomma(f, k, tau, sig, b)
        vega = Black76Vec.vega(f, k, tau, sig, b)
        d_black_d_k = model.calc_d_black_d_k(f, k)
        d2_black_d_k2 = model.calc_d2_black_d_k2(f, k)
        return gamma_k + 2 * vanna * d_black_d_k + vomma * (d_black_d_k ** 2) + vega * d2_black_d_k2
