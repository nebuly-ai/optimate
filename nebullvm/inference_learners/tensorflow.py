import os
import pickle
import shutil
from pathlib import Path
from typing import Tuple, Union, Dict, Type

from nebullvm.base import ModelParams, Device
from nebullvm.config import TENSORFLOW_BACKEND_FILENAMES
from nebullvm.inference_learners import (
    TensorflowBaseInferenceLearner,
    LearnerMetadata,
)
from nebullvm.optional_modules.tensorflow import tensorflow as tf


class TensorflowBackendInferenceLearner(TensorflowBaseInferenceLearner):
    def __init__(self, tf_model: tf.Module, device: Device, **kwargs):
        super(TensorflowBackendInferenceLearner, self).__init__(**kwargs)
        self.model = tf_model
        self.device = device

    def get_size(self):
        return len(pickle.dumps(self.model, -1))

    def run(self, *input_tensors: tf.Tensor) -> Tuple[tf.Tensor, ...]:
        with tf.device(self.device.value):
            res = self.model.predict(input_tensors)
        if not isinstance(res, tuple):
            return (res,)
        return res

    def save(self, path: Union[str, Path], **kwargs):
        path = Path(path)
        path.mkdir(exist_ok=True)
        metadata = LearnerMetadata.from_model(self, **kwargs)
        metadata.save(path)
        self.model.save(path / TENSORFLOW_BACKEND_FILENAMES["tf_model"])

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        path = Path(path)
        metadata = LearnerMetadata.read(path)
        network_parameters = ModelParams(**metadata.network_parameters)
        input_tfms = metadata.input_tfms
        model = tf.keras.models.load_model(
            path / TENSORFLOW_BACKEND_FILENAMES["tf_model"]
        )
        device = metadata.device
        return cls(
            tf_model=model,
            network_parameters=network_parameters,
            input_tfms=input_tfms,
            device=device,
        )


class TFLiteBackendInferenceLearner(TensorflowBaseInferenceLearner):
    def __init__(self, tflite_file: str, device: Device, **kwargs):
        super(TFLiteBackendInferenceLearner, self).__init__(**kwargs)
        self._tflite_file = self._store_file(tflite_file)
        self.interpreter = tf.lite.Interpreter(tflite_file)
        self.device = device

    def get_size(self):
        return os.path.getsize(self._tflite_file)

    def run(self, *input_tensors: tf.Tensor):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        if self.network_parameters.dynamic_info:
            for i, (input_tensor, detail) in enumerate(
                zip(input_tensors, input_details)
            ):
                if input_tensor.shape != detail["shape"]:
                    self.interpreter.resize_tensor_input(i, input_tensor.shape)
        self.interpreter.allocate_tensors()
        for i, input_tensor in enumerate(input_tensors):
            self.interpreter.set_tensor(i, input_tensor)
        self.interpreter.invoke()
        return tuple(
            self.interpreter.get_tensor(output_detail["index"])
            for output_detail in output_details
        )

    def save(self, path: Union[str, Path], **kwargs):
        path = Path(path)
        metadata = LearnerMetadata.from_model(self, **kwargs)
        metadata.save(path)
        shutil.copy2(
            self._tflite_file,
            path / TENSORFLOW_BACKEND_FILENAMES["tflite_model"],
        )

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        path = Path(path)
        tflite_file = str(path / TENSORFLOW_BACKEND_FILENAMES["tflite_model"])
        metadata = LearnerMetadata.read(path)
        network_parameters = ModelParams(**metadata.network_parameters)
        input_tfms = metadata.input_tfms
        device = metadata.device
        return cls(
            tflite_file=tflite_file,
            network_parameters=network_parameters,
            input_tfms=input_tfms,
            device=device,
        )


TF_BACKEND_LEARNERS_DICT: Dict[
    str,
    Type[
        Union[TensorflowBackendInferenceLearner, TFLiteBackendInferenceLearner]
    ],
] = {
    "tf": TensorflowBackendInferenceLearner,
    "tflite": TFLiteBackendInferenceLearner,
}
