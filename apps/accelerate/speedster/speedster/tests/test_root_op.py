from nebullvm.core.models import OptimizeInferenceResult

from speedster.root_op import SpeedsterRootOp


def test_root_op_no_optim_model(mocker):
    root_op = SpeedsterRootOp()

    mocker.patch.object(
        root_op.optimize_inference_op,
        "execute",
        return_value=OptimizeInferenceResult(
            original_model=mocker.MagicMock(),
            optimized_model=None,
            hardware_setup=mocker.MagicMock(),
        ),
    )

    res = root_op.execute(
        model=None,
        input_data=mocker.MagicMock(),
        metric_drop_ths=None,
        metric="latency",
        optimization_time=mocker.MagicMock(),
        dynamic_info=None,
        config_file=None,
        ignore_compilers=None,
        ignore_compressors=None,
        store_latencies=False,
    )

    assert res is None


def test_root_op_optim_model(mocker):
    root_op = SpeedsterRootOp()

    mocker.patch.object(
        root_op.optimize_inference_op,
        "execute",
        return_value=OptimizeInferenceResult(
            original_model=mocker.MagicMock(
                latency_seconds=1, throughput=1, size_mb=1
            ),
            optimized_model=mocker.MagicMock(
                metric_drop=0.1, latency_seconds=1, size_mb=1, throughput=1
            ),
            hardware_setup=mocker.MagicMock(),
        ),
    )

    mocker.patch.object(root_op, "_send_feedback")

    res = root_op.execute(
        model=None,
        input_data=mocker.MagicMock(),
        metric_drop_ths=None,
        metric="latency",
        optimization_time=mocker.MagicMock(),
        dynamic_info=None,
        config_file=None,
        ignore_compilers=None,
        ignore_compressors=None,
        store_latencies=False,
    )

    assert res is not None
