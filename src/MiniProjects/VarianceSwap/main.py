import numpy as np

from src.MiniProjects.VarianceSwap.VarianceReplication import VarianceReplication
from src.Utils.Black76 import Black76Vec
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'

tau = 5.0  # 30.0
b = 0.87
fwd = 150.0
flat_vols = np.linspace(0.1, 1.5, num=15)

for flat_vol in flat_vols:
    scale = fwd * flat_vol * np.sqrt(tau)
    strikes = np.linspace(0.0001 * scale, 20.0 * scale, num=10 ** 6)
    prices_call = Black76Vec.price(fwd, strikes, tau, flat_vol, b, OptionType.call)
    prices_put = Black76Vec.price(fwd, strikes, tau, flat_vol, b, OptionType.put)

    rp = VarianceReplication(tau, fwd, b, strikes, prices_put, strikes, prices_call)
    var = rp.calc_variance()
    std = np.sqrt(var)
    print('%.2f\t%.12f\t%.4e' % (flat_vol, std, float(std / flat_vol - 1.0)))