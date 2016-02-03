import logging

import numpy as np
import scipy.optimize as opt
from scipy import interp

from src.Utils.Valuator.CashFlowDiscounter import CashFlowDiscounter

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class MarketRateToZeroRate(object):
    def __init__(self, bonds: list, weights: list, tenors: np.array, interp_method: str = 'linear'):
        for bond in bonds:
            if not isinstance(bond, CashFlowDiscounter):
                msg = 'input bond (%s) is not an instance of %s' % bond.__name__, CashFlowDiscounter.__name__
                logger.error(msg)
                raise TypeError(msg)
        self.bonds = bonds
        self.weights = np.array(weights) / np.sum(weights)  # normalized weights to one
        self.market_rates = [bond.calc_irr() for bond in bonds]
        self.bonds_maturities = [max(bond.ts) for bond in bonds]
        self.tenors = np.sort(tenors)
        self.interp_method = interp_method
        pass

    def fitting_error_function(self, zero_rates: tuple) -> float:
        errors = np.full(self.bonds.__len__(), 1.0)  # initialize npv errors as ones
        for idx, bond in enumerate(self.bonds):
            rates = interp(bond.ts, self.tenors, zero_rates)
            npv = bond.calc_npv(rates)
            errors[idx] = npv
        weighted_errors_sq = np.power(errors * self.weights, 2)
        return sum(weighted_errors_sq)

    def fit_zero_curve(self):
        # initial guess is on market rate
        init_guess = np.interp(self.tenors, self.bonds_maturities, self.market_rates)
        bounds = [(-0.05, 1.0) for _ in range(init_guess.__len__())]
        res = opt.minimize(self.fitting_error_function, init_guess, method='L-BFGS-B', jac=False,
                           bounds=bounds, tol=1e-8)
        zero_rates = res.x
        return zero_rates

    def bootstrap(self):
        bonds_union = dict(zip(self.bonds_maturities, self.bonds))
        tenors = []
        zero_rates = []
        for idx, bond_mat in enumerate(sorted(bonds_union.keys())):
            bond = bonds_union[bond_mat]

            if idx == 0:
                tenors += list(bond.ts)
                zero_rates += [bond.calc_irr for _ in range(tenors.__len__())]
            else:
                tenors += []
                zero_rates += []

        return dict(zip(tenors, zero_rates))
