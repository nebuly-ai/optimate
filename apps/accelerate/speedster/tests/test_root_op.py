import unittest
from unittest.mock import MagicMock

from speedster.root_op import SpeedsterRootOp


class TestSpeedsterRootOp(unittest.TestCase):
    def test_execute__empty_input_data_should_raise_error(self):
        op = SpeedsterRootOp()
        with self.assertRaises(ValueError):
            op.execute(
                model=MagicMock(),
                input_data=[],
            )

    def test_execute__none_model_should_raise_error(self):
        op = SpeedsterRootOp()
        with self.assertRaises(ValueError):
            op.execute(
                model=None,
                input_data=[i for i in range(10)],
            )
