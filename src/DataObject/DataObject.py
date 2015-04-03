import datetime

__author__ = 'frank.ma'


class DataObject(object):

    def __init__(self, asof: datetime):
        self.__asof = asof  # make local variable to avoid accidental updates

    def get_asof(self):
        return self.__asof