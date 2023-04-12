import unittest
from unittest.mock import patch

from nebullvm.tools import hardware_utils


class TestGetHwSetup(unittest.TestCase):
    @patch(
        "nebullvm.tools.hardware_utils.gpu_is_available", return_value=False
    )
    @patch(
        "nebullvm.tools.hardware_utils.tpu_is_available", return_value=False
    )
    @patch(
        "nebullvm.tools.hardware_utils.neuron_is_available", return_value=False
    )
    def test_hw_setup__gpu_not_available(self, *_):
        setup = hardware_utils.get_hw_setup()
        self.assertIsNone(setup.accelerator)
        self.assertGreater(len(setup.cpu), 0)
        self.assertGreater(len(setup.operating_system), 0)
        self.assertGreater(setup.memory_gb, 0)

    @patch("nebullvm.tools.hardware_utils.gpu_is_available", return_value=True)
    @patch(
        "nebullvm.tools.hardware_utils._get_gpu_name", return_value="mock-gpu"
    )
    def test_hw_setup__gpu_is_available(self, *_):
        setup = hardware_utils.get_hw_setup()
        self.assertEqual("mock-gpu", setup.accelerator)
        self.assertGreater(len(setup.cpu), 0)
        self.assertGreater(len(setup.operating_system), 0)
        self.assertGreater(setup.memory_gb, 0)
