from tempfile import TemporaryDirectory

import pytest

from nebullvm.base import QuantizationType
from nebullvm.optimizers.tests.utils import get_onnx_model
from nebullvm.quantizers.onnx_quantizer import (
    ONNX_QUANTIZER_DICT,
    ONNXQuantizerManager,
)
from nebullvm.utils.onnx import create_model_inputs_onnx


@pytest.mark.parametrize(
    "q_type", [QuantizationType.STATIC, QuantizationType.DYNAMIC]
)
def test_onnx_quantizer(q_type: QuantizationType):
    with TemporaryDirectory() as tmp_dir:
        quantizer = ONNX_QUANTIZER_DICT[q_type](tolerated_error=1)
        onnx_model, model_params = get_onnx_model(tmp_dir)
        input_data = [
            tuple(
                create_model_inputs_onnx(
                    model_params.batch_size, model_params.input_infos
                )
            )
        ]
        quantized_onnx = quantizer(onnx_model, input_data=input_data)
        assert len(quantized_onnx) > 0


@pytest.mark.parametrize(
    "q_type", [QuantizationType.STATIC, QuantizationType.DYNAMIC, None]
)
def test_quantizer_manager(q_type: QuantizationType):
    quantizer_manager = ONNXQuantizerManager(tolerated_error=1)
    with TemporaryDirectory() as tmp_dir:
        onnx_model, model_params = get_onnx_model(tmp_dir)
        quantized_model = quantizer_manager.run(
            onnx_model, model_params, quantization_type=q_type
        )
        assert quantized_model is None or len(quantized_model) > 0
