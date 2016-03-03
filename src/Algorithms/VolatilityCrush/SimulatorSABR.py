import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from src.SABRModel.SABRModel import SABRModelLognormalApprox
from src.Utils.OptionType import OptionType
from src.Utils.Valuator.Black76 import Black76Vec
from src.Utils.Valuator.VarianceReplication import VarianceReplication

__author__ = 'frank.ma'


class SimulatorSABR(object):
    ONE_DAY = 1.0 / 365.25

    def __init__(self, model: SABRModelLognormalApprox, forward: float, bond: float):
        self.model_pre = model
        self.forward_pre = forward
        self.bond_pre = bond
        pass

    @staticmethod
    def model_shock(model: SABRModelLognormalApprox, dt: float = ONE_DAY, alpha_shift: float = 1.0,
                    nu_shift: float = 1.0, rho_shift: float = 1.0):
        t = model.t - dt
        alpha = model.alpha * alpha_shift
        beta = model.beta  # do not expect beta to be shocked
        nu = model.nu * nu_shift
        rho = model.rho * rho_shift
        return SABRModelLognormalApprox(t, alpha, beta, nu, rho)

    def display_vol_shocks(self, dt: float = ONE_DAY, alpha_shift: float = 1.0, nu_shift: float = 1.0,
                           rho_shift: float = 1.0, n_scenarios: int = 5):
        # pre event volatility
        ks = np.linspace(0.01 * self.forward_pre, 20.0 * self.forward_pre, num=4999)
        vs = self.model_pre.calc_vol_vec(self.forward_pre, ks)
        v_c = Black76Vec.price(self.forward_pre, ks, self.model_pre.t, vs, self.bond_pre, OptionType.call)
        v_p = Black76Vec.price(self.forward_pre, ks, self.model_pre.t, vs, self.bond_pre, OptionType.put)
        var = VarianceReplication(self.model_pre.t, self.forward_pre, self.bond_pre, ks, v_p, ks, v_c).calc_variance()
        vol_rep = np.sqrt(var)
        vol_atm = self.model_pre.calc_vol(self.forward_pre, self.forward_pre)

        # post event volatility
        model_post = self.model_shock(self.model_pre, dt, alpha_shift, nu_shift, rho_shift)
        b_use_rep = True
        vol = vol_rep if b_use_rep else vol_atm
        rand = np.random.random(n_scenarios)
        forwards_post = self.forward_pre * np.exp(vol * np.sqrt(dt) * norm.ppf(rand))

        ks_display = np.linspace(0.5 * self.forward_pre, 1.5 * self.forward_pre, num=50)
        vols_pre = self.model_pre.calc_vol_vec(self.forward_pre, ks_display)
        plt.plot(ks_display, vols_pre, label='pre_event')
        for forward_post in forwards_post:
            vols_post = model_post.calc_vol_vec(forward_post, ks_display)
            plt.plot(ks_display, vols_post, label='%.2f' % forward_post)
        plt.legend()
        plt.show()
        pass
