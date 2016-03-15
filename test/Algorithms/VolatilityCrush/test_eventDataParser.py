from unittest import TestCase

from src.Algorithms.VolatilityCrush.EventDataParser import EventDataParser

__author__ = 'frank.ma'


class TestEventDataParser(TestCase):
    def test_read_events_from_csv(self):
        path = './Data/sample_2w_pre.csv'
        EventDataParser.read_events_from_csv(path)
        pass
