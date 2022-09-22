import torch

from nebullvm.api.utils import QUANTIZATION_METRIC_MAP
from nebullvm.base import DeepLearningFramework
from nebullvm.compressors.sparseml import SparseMLCompressor
from nebullvm.optimizers.tests.utils import initialize_model


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
        metric_drop_ths=None,
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
