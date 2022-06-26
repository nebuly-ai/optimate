from typing import List, Tuple

import tensorflow as tf

from nebullvm.base import QuantizationType
from nebullvm.transformations.base import MultiStageTransformation


def _quantize_dynamic(model: tf.Module):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    return tflite_quant_model


def _quantize_static(model: tf.Module, dataset: List[Tuple[tf.Tensor, ...]]):
    def representative_dataset():
        for data_tuple in dataset:
            yield list(data_tuple)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    tflite_quant_model = converter.convert()
    return tflite_quant_model


def _half_precision(model: tf.Module):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_quant_model = converter.convert()
    return tflite_quant_model


def quantize_tf(
    model: tf.Module,
    quantization_type: QuantizationType,
    input_tfms: MultiStageTransformation,
    input_data_torch: List[Tuple[tf.Tensor, ...]],
):
    if quantization_type is QuantizationType.DYNAMIC:
        return _quantize_dynamic(model), input_tfms
    elif quantization_type is QuantizationType.STATIC:
        return _quantize_static(model, input_data_torch), input_tfms
    elif quantization_type is QuantizationType.HALF:
        return _half_precision(model), input_tfms
    else:
        raise NotImplementedError(
            f"Quantization not supported for type {quantization_type}"
        )
