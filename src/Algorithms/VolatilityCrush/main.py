from datetime import datetime, timedelta

__author__ = 'frank.ma'

dt_pre = datetime(2016, 2, 16, 16, 0, 0)
dt_event = datetime(2016, 2, 17, 16, 0, 0)
dt_post = datetime(2016, 2, 18, 16, 0, 0)

exp_s = datetime(2016, 2, 19)
exp_l = datetime(2016, 3, 18)

tau_s = (exp_s - dt_pre) / timedelta(365.25)
tau_l = (exp_l - dt_pre) / timedelta(365.25)

print(tau_s, tau_l)

r = 0.05
q = 0.03

sig_s = 0.55
sig_l = 0.45

strike = 155.0
spot_pre = 150.0
spot_post = 170.0

