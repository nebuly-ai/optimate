import cpuinfo

import pytest
import torch

from nebullvm.api.utils import QUANTIZATION_METRIC_MAP
from nebullvm.base import DeepLearningFramework
from nebullvm.compressors.intel import TorchIntelPruningCompressor
from nebullvm.optimizers.tests.utils import initialize_model


@pytest.mark.skipif(
    "intel" in cpuinfo.get_cpu_info()["brand_raw"].lower(),
    reason="Tested only on intel cpus.",
)
def test_intel_pruning():
    (
        model,
        input_data,
        model_params,
        input_tfms,
        model_outputs,
        metric,
    ) = initialize_model(
        dynamic=False,
        metric_drop_ths=None,
        metric=None,
        output_library=DeepLearningFramework.PYTORCH,
    )

    compressor = TorchIntelPruningCompressor()

    pruned_model, new_metric_ths = compressor.compress(
        model,
        input_data,
        input_data,
        2,
        QUANTIZATION_METRIC_MAP["numeric_precision"],
    )

    assert isinstance(pruned_model, torch.nn.Module)
    assert new_metric_ths <= 2
