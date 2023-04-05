import unittest

from nebullvm.tools import utils


class TestGetThroughput(unittest.TestCase):
    def test_latency_is_zero(self):
        self.assertEqual(-1, utils.get_throughput(0, 10))