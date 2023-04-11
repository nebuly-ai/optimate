import unittest
from unittest.mock import MagicMock

from nebullvm.core.models import OptimizeInferenceResult


class TestOptimizeInferenceResult(unittest.TestCase):
    def test_latency_improvement_rate__optimized_model_is_none(self):
        res = OptimizeInferenceResult(
            original_model=MagicMock(),
            hardware_setup=MagicMock(),
            optimized_model=None,
        )
        self.assertIsNone(res.latency_improvement_rate)

    def test_latency_improvement_rate__optimized_latency_is_zero(self):
        original_latency = 1.0
        optimized_latency = 0.0
        res = OptimizeInferenceResult(
            original_model=MagicMock(latency_seconds=original_latency),
            hardware_setup=MagicMock(),
            optimized_model=MagicMock(latency_seconds=optimized_latency),
        )
        self.assertEqual(-1, res.latency_improvement_rate)

    def test_latency_improvement_rate__original_latency_is_zero(self):
        original_latency = 0.0
        optimized_latency = 1.0
        res = OptimizeInferenceResult(
            original_model=MagicMock(latency_seconds=original_latency),
            hardware_setup=MagicMock(),
            optimized_model=MagicMock(latency_seconds=optimized_latency),
        )
        self.assertEqual(0, res.latency_improvement_rate)

    def test_latency_improvement_rate__rate_gt_1(self):
        original_latency = 1.0
        optimized_latency = 0.5
        res = OptimizeInferenceResult(
            original_model=MagicMock(latency_seconds=original_latency),
            hardware_setup=MagicMock(),
            optimized_model=MagicMock(latency_seconds=optimized_latency),
        )
        self.assertGreater(res.latency_improvement_rate, 1)

    def test_latency_improvement_rate__rate_lt_1(self):
        original_latency = 0.5
        optimized_latency = 1.0
        res = OptimizeInferenceResult(
            original_model=MagicMock(latency_seconds=original_latency),
            hardware_setup=MagicMock(),
            optimized_model=MagicMock(latency_seconds=optimized_latency),
        )
        self.assertLess(res.latency_improvement_rate, 1)

    def test_th_improvement_rate__optimized_model_is_none(self):
        res = OptimizeInferenceResult(
            original_model=MagicMock(),
            hardware_setup=MagicMock(),
            optimized_model=None,
        )
        self.assertIsNone(res.throughput_improvement_rate)

    def test_th_improvement_rate__optimized_th_is_zero(self):
        original_th = 1.0
        optimized_th = 0.0
        res = OptimizeInferenceResult(
            original_model=MagicMock(throughput=original_th),
            hardware_setup=MagicMock(),
            optimized_model=MagicMock(throughput=optimized_th),
        )
        self.assertEqual(0, res.throughput_improvement_rate)

    def test_th_improvement_rate__original_th_is_zero(self):
        original_th = 0.0
        optimized_th = 1.0
        res = OptimizeInferenceResult(
            original_model=MagicMock(throughput=original_th),
            hardware_setup=MagicMock(),
            optimized_model=MagicMock(throughput=optimized_th),
        )
        self.assertEqual(-1, res.throughput_improvement_rate)

    def test_th_improvement_rate__rate_gt_1(self):
        original_th = 0.5
        optimized_th = 1
        res = OptimizeInferenceResult(
            original_model=MagicMock(throughput=original_th),
            hardware_setup=MagicMock(),
            optimized_model=MagicMock(throughput=optimized_th),
        )
        self.assertGreater(res.throughput_improvement_rate, 1)

    def test_th_improvement_rate__rate_lt_1(self):
        original_th = 1.0
        optimized_th = 0.5
        res = OptimizeInferenceResult(
            original_model=MagicMock(throughput=original_th),
            hardware_setup=MagicMock(),
            optimized_model=MagicMock(throughput=optimized_th),
        )
        self.assertLess(res.throughput_improvement_rate, 1)

    def test_size_improvement_rate__optimized_model_is_none(self):
        res = OptimizeInferenceResult(
            original_model=MagicMock(),
            hardware_setup=MagicMock(),
            optimized_model=None,
        )
        self.assertIsNone(res.size_improvement_rate)

    def test_size_improvement_rate__optimized_size_is_zero(self):
        original_size = 1.0
        optimized_size = 0.0
        res = OptimizeInferenceResult(
            original_model=MagicMock(size_mb=original_size),
            hardware_setup=MagicMock(),
            optimized_model=MagicMock(size_mb=optimized_size),
        )
        self.assertEqual(1, res.size_improvement_rate)

    def test_size_improvement_rate__original_size_is_zero(self):
        original_size = 0.0
        optimized_size = 1.0
        res = OptimizeInferenceResult(
            original_model=MagicMock(size_mb=original_size),
            hardware_setup=MagicMock(),
            optimized_model=MagicMock(size_mb=optimized_size),
        )
        self.assertEqual(0, res.size_improvement_rate)

    def test_size_improvement_rate__rate_gt_1(self):
        original_size = 1
        optimized_size = 0.5
        res = OptimizeInferenceResult(
            original_model=MagicMock(size_mb=original_size),
            hardware_setup=MagicMock(),
            optimized_model=MagicMock(size_mb=optimized_size),
        )
        self.assertGreater(res.size_improvement_rate, 1)

    def test_size_improvement_rate__rate_lt_1(self):
        original_size = 0.5
        optimized_size = 1
        res = OptimizeInferenceResult(
            original_model=MagicMock(size_mb=original_size),
            hardware_setup=MagicMock(),
            optimized_model=MagicMock(size_mb=optimized_size),
        )
        self.assertLess(res.size_improvement_rate, 1)

    def test_metric_drop__optimized_model_is_none(self):
        res = OptimizeInferenceResult(
            original_model=MagicMock(),
            hardware_setup=MagicMock(),
            optimized_model=None,
        )
        self.assertIsNone(res.metric_drop)

    def test_metric_drop(self):
        metric_drop = 0.1
        res = OptimizeInferenceResult(
            original_model=MagicMock(),
            hardware_setup=MagicMock(),
            optimized_model=MagicMock(metric_drop=metric_drop),
        )
        self.assertEqual(metric_drop, res.metric_drop)
