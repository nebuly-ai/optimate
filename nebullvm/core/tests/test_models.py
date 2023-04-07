from unittest.mock import MagicMock

from nebullvm.core.models import OptimizeInferenceResult


def test_latency_improvement_rate__optimized_model_is_none():
    res = OptimizeInferenceResult(
        original_model=MagicMock(),
        hardware_setup=MagicMock(),
        optimized_model=None,
    )
    assert res.latency_improvement_rate is None


def test_latency_improvement_rate__optimized_latency_is_zero():
    original_latency = 1.0
    optimized_latency = 0.0
    res = OptimizeInferenceResult(
        original_model=MagicMock(latency_seconds=original_latency),
        hardware_setup=MagicMock(),
        optimized_model=MagicMock(latency_seconds=optimized_latency),
    )
    assert res.latency_improvement_rate == -1


def test_latency_improvement_rate__original_latency_is_zero():
    original_latency = 0.0
    optimized_latency = 1.0
    res = OptimizeInferenceResult(
        original_model=MagicMock(latency_seconds=original_latency),
        hardware_setup=MagicMock(),
        optimized_model=MagicMock(latency_seconds=optimized_latency),
    )
    assert res.latency_improvement_rate == 0


def test_latency_improvement_rate__rate_gt_1():
    original_latency = 1.0
    optimized_latency = 0.5
    res = OptimizeInferenceResult(
        original_model=MagicMock(latency_seconds=original_latency),
        hardware_setup=MagicMock(),
        optimized_model=MagicMock(latency_seconds=optimized_latency),
    )
    assert res.latency_improvement_rate > 1


def test_latency_improvement_rate__rate_lt_1():
    original_latency = 0.5
    optimized_latency = 1.0
    res = OptimizeInferenceResult(
        original_model=MagicMock(latency_seconds=original_latency),
        hardware_setup=MagicMock(),
        optimized_model=MagicMock(latency_seconds=optimized_latency),
    )
    assert res.latency_improvement_rate < 1


def test_th_improvement_rate__optimized_model_is_none():
    res = OptimizeInferenceResult(
        original_model=MagicMock(),
        hardware_setup=MagicMock(),
        optimized_model=None,
    )

    assert res.throughput_improvement_rate is None


def test_th_improvement_rate__optimized_th_is_zero():
    original_th = 1.0
    optimized_th = 0.0
    res = OptimizeInferenceResult(
        original_model=MagicMock(throughput=original_th),
        hardware_setup=MagicMock(),
        optimized_model=MagicMock(throughput=optimized_th),
    )
    assert res.throughput_improvement_rate == 0


def test_th_improvement_rate__original_th_is_zero():
    original_th = 0.0
    optimized_th = 1.0
    res = OptimizeInferenceResult(
        original_model=MagicMock(throughput=original_th),
        hardware_setup=MagicMock(),
        optimized_model=MagicMock(throughput=optimized_th),
    )
    assert res.throughput_improvement_rate == -1


def test_th_improvement_rate__rate_gt_1():
    original_th = 0.5
    optimized_th = 1
    res = OptimizeInferenceResult(
        original_model=MagicMock(throughput=original_th),
        hardware_setup=MagicMock(),
        optimized_model=MagicMock(throughput=optimized_th),
    )
    assert res.throughput_improvement_rate > 1


def test_th_improvement_rate__rate_lt_1():
    original_th = 1.0
    optimized_th = 0.5
    res = OptimizeInferenceResult(
        original_model=MagicMock(throughput=original_th),
        hardware_setup=MagicMock(),
        optimized_model=MagicMock(throughput=optimized_th),
    )
    assert res.throughput_improvement_rate < 1


def test_size_improvement_rate__optimized_model_is_none():
    res = OptimizeInferenceResult(
        original_model=MagicMock(),
        hardware_setup=MagicMock(),
        optimized_model=None,
    )
    assert res.size_improvement_rate is None


def test_size_improvement_rate__optimized_size_is_zero():
    original_size = 1.0
    optimized_size = 0.0
    res = OptimizeInferenceResult(
        original_model=MagicMock(size_mb=original_size),
        hardware_setup=MagicMock(),
        optimized_model=MagicMock(size_mb=optimized_size),
    )
    assert res.size_improvement_rate == 1


def test_size_improvement_rate__original_size_is_zero():
    original_size = 0.0
    optimized_size = 1.0
    res = OptimizeInferenceResult(
        original_model=MagicMock(size_mb=original_size),
        hardware_setup=MagicMock(),
        optimized_model=MagicMock(size_mb=optimized_size),
    )
    assert res.size_improvement_rate == 0


def test_size_improvement_rate__rate_gt_1():
    original_size = 1
    optimized_size = 0.5
    res = OptimizeInferenceResult(
        original_model=MagicMock(size_mb=original_size),
        hardware_setup=MagicMock(),
        optimized_model=MagicMock(size_mb=optimized_size),
    )
    assert res.size_improvement_rate > 1


def test_size_improvement_rate__rate_lt_1():
    original_size = 0.5
    optimized_size = 1
    res = OptimizeInferenceResult(
        original_model=MagicMock(size_mb=original_size),
        hardware_setup=MagicMock(),
        optimized_model=MagicMock(size_mb=optimized_size),
    )
    assert res.size_improvement_rate < 1


def test_metric_drop__optimized_model_is_none():
    res = OptimizeInferenceResult(
        original_model=MagicMock(),
        hardware_setup=MagicMock(),
        optimized_model=None,
    )
    assert res.metric_drop is None


def test_metric_drop():
    metric_drop = 0.1
    res = OptimizeInferenceResult(
        original_model=MagicMock(),
        hardware_setup=MagicMock(),
        optimized_model=MagicMock(metric_drop=metric_drop),
    )
    assert metric_drop == res.metric_drop
