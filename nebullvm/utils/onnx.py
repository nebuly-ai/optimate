from typing import List

import numpy as np
import onnx

from nebullvm.base import InputInfo, DataType


def get_input_names(onnx_model: str):
    model = onnx.load(onnx_model)

    input_all = [node.name for node in model.graph.input]
    return input_all


def get_output_names(onnx_model: str):
    model = onnx.load(onnx_model)
    output_all = [node.name for node in model.graph.output]
    return output_all


def get_output_sizes_onnx(onnx_model: str, input_tensors: List[np.ndarray]):
    import onnxruntime as ort

    model = ort.InferenceSession(
        onnx_model, providers=["CPUExecutionProvider"]
    )
    inputs = {
        name: array
        for name, array in zip(get_input_names(onnx_model), input_tensors)
    }
    res = model.run(
        output_names=get_output_names(onnx_model), input_feed=inputs
    )
    sizes = [tuple(output.shape[1:]) for output in res]
    return sizes


def create_model_inputs_onnx(
    batch_size: int, input_infos: List[InputInfo]
) -> List[np.ndarray]:
    input_tensors = (
        np.random.randn(batch_size, *input_info.size).astype(np.float32)
        if input_info.dtype is DataType.FLOAT
        else np.random.randint(
            size=(batch_size, *input_info.size),
            low=input_info.min_value or 0,
            high=input_info.max_value or 100,
        )
        for input_info in input_infos
    )
    return list(input_tensors)
