from xlwings import Workbook, Sheet, Range, Chart
import time

__author__ = 'frank.ma'

wb = Workbook()


def get_quote(ticker, type):
    Range('B1').value = '=RTD("tos.rtd", , "%s", "%s")' % (type, ticker)
    time.sleep(2.5)  # sleep for 2.5 second
    return Range('B1').value


a = get_quote('/ESU5', 'LAST')
print(a)
