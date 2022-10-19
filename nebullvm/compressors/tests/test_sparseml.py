import pytest
import torch

from nebullvm.api.utils import QUANTIZATION_METRIC_MAP
from nebullvm.base import DeepLearningFramework
from nebullvm.compressors.sparseml import SparseMLCompressor
from nebullvm.optimizers.tests.utils import initialize_model
from nebullvm.utils.general import is_python_version_3_10


@pytest.mark.skipif(
    is_python_version_3_10(),
    reason="Torch 1.9.1 is not available in python 1.10",
)
def test_sparseml():
    (
        model,
        input_data,
        model_params,
        input_tfms,
        model_outputs,
        metric,
    ) = initialize_model(
        dynamic=False,
        metric=None,
        output_library=DeepLearningFramework.PYTORCH,
    )

    compressor = SparseMLCompressor()

    pruned_model, new_metric_ths = compressor.compress(
        model,
        input_data,
        input_data,
        2,
        QUANTIZATION_METRIC_MAP["numeric_precision"],
    )

    assert isinstance(pruned_model, torch.nn.Module) or isinstance(
        pruned_model, torch.fx.GraphModule
    )
    assert new_metric_ths <= 2
