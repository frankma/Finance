from datetime import datetime, timedelta

__author__ = 'frank.ma'

asof = datetime(2016, 2, 12)
event = datetime(2016, 2, 17)

exp_s = datetime(2016, 2, 19)
exp_l = datetime(2016, 3, 18)

tau_s = (exp_s - asof) / timedelta(365.25)
tau_l = (exp_l - asof) / timedelta(365.25)
print(tau_s, tau_l)
