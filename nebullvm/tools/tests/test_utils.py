import unittest
from unittest.mock import patch

from nebullvm.core.models import DeviceType
from nebullvm.tools import utils


class TestGetThroughput(unittest.TestCase):
    def test_latency_is_zero(self):
        self.assertEqual(-1, utils.get_throughput(0, 10))


class TestCheckDevice(unittest.TestCase):
    @patch("nebullvm.tools.utils.gpu_is_available", return_value=False)
    @patch("nebullvm.tools.utils.tpu_is_available", return_value=False)
    @patch("nebullvm.tools.utils.neuron_is_available", return_value=False)
    def test_device_is_none_no_device_available(self, *_):
        device = utils.check_device()
        self.assertEqual(DeviceType.CPU, device.type)
        self.assertEqual(device.idx, 0)

    @patch("nebullvm.tools.utils.gpu_is_available", return_value=True)
    @patch("nebullvm.tools.utils.neuron_is_available", return_value=False)
    @patch("nebullvm.tools.utils.tpu_is_available", return_value=False)
    def test_device_is_none_gpu_is_available(self, *_):
        device = utils.check_device()
        self.assertEqual(DeviceType.GPU, device.type)
        self.assertEqual(device.idx, 0)

    @patch("nebullvm.tools.utils.tpu_is_available", return_value=True)
    @patch("nebullvm.tools.utils.gpu_is_available", return_value=False)
    @patch("nebullvm.tools.utils.neuron_is_available", return_value=False)
    def test_device_is_none_tpu_is_available(self, *_):
        device = utils.check_device()
        self.assertEqual(DeviceType.TPU, device.type)
        self.assertEqual(device.idx, 0)

    @patch("nebullvm.tools.utils.neuron_is_available", return_value=True)
    @patch("nebullvm.tools.utils.gpu_is_available", return_value=False)
    @patch("nebullvm.tools.utils.tpu_is_available", return_value=False)
    def test_device_is_none_neuron_is_available(self, *_):
        device = utils.check_device()
        self.assertEqual(DeviceType.NEURON, device.type)
        self.assertEqual(device.idx, 0)

    def test_device_is_cpu(self):
        device = utils.check_device("cpu")
        self.assertEqual(DeviceType.CPU, device.type)
        self.assertEqual(device.idx, 0)

    @patch("nebullvm.tools.utils.gpu_is_available", return_value=False)
    def test_device_is_gpu_no_gpu_available(self, _):
        device = utils.check_device("gpu")
        self.assertEqual(DeviceType.CPU, device.type)
        self.assertEqual(device.idx, 0)

        device = utils.check_device("cuda")
        self.assertEqual(DeviceType.CPU, device.type)
        self.assertEqual(device.idx, 0)

        device = utils.check_device("cuda:1")
        self.assertEqual(DeviceType.CPU, device.type)
        self.assertEqual(device.idx, 0)

        device = utils.check_device("gpu:2")
        self.assertEqual(DeviceType.CPU, device.type)
        self.assertEqual(device.idx, 0)

    @patch("nebullvm.tools.utils.gpu_is_available", return_value=True)
    def test_device_is_gpu_gpu_is_available(self, _):
        device = utils.check_device("gpu")
        self.assertEqual(DeviceType.GPU, device.type)
        self.assertEqual(device.idx, 0)

        device = utils.check_device("cuda")
        self.assertEqual(DeviceType.GPU, device.type)
        self.assertEqual(device.idx, 0)

        device = utils.check_device("cuda:1")
        self.assertEqual(DeviceType.GPU, device.type)
        self.assertEqual(device.idx, 1)

        device = utils.check_device("gpu:2")
        self.assertEqual(DeviceType.GPU, device.type)
        self.assertEqual(device.idx, 2)

    @patch("nebullvm.tools.utils.tpu_is_available", return_value=False)
    def test_device_is_tpu_no_tpu_available(self, _):
        device = utils.check_device("tpu")
        self.assertEqual(DeviceType.CPU, device.type)
        self.assertEqual(device.idx, 0)

        device = utils.check_device("tpu:1")
        self.assertEqual(DeviceType.CPU, device.type)
        self.assertEqual(device.idx, 0)

    @patch("nebullvm.tools.utils.tpu_is_available", return_value=True)
    def test_device_is_tpu_tpu_is_available(self, _):
        device = utils.check_device("tpu")
        self.assertEqual(DeviceType.TPU, device.type)
        self.assertEqual(device.idx, 0)

        device = utils.check_device("tpu:1")
        self.assertEqual(DeviceType.TPU, device.type)
        self.assertEqual(device.idx, 1)

    @patch("nebullvm.tools.utils.neuron_is_available", return_value=False)
    def test_device_is_neuron_no_neuron_available(self, _):
        device = utils.check_device("neuron")
        self.assertEqual(DeviceType.CPU, device.type)
        self.assertEqual(device.idx, 0)

        device = utils.check_device("neuron:1")
        self.assertEqual(DeviceType.CPU, device.type)
        self.assertEqual(device.idx, 0)

    @patch("nebullvm.tools.utils.neuron_is_available", return_value=True)
    def test_device_is_neuron_neuron_is_available(self, _):
        device = utils.check_device("neuron")
        self.assertEqual(DeviceType.NEURON, device.type)
        self.assertEqual(device.idx, 0)

        device = utils.check_device("neuron:1")
        self.assertEqual(DeviceType.NEURON, device.type)
        self.assertEqual(device.idx, 1)
