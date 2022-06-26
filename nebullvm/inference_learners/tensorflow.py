from pathlib import Path
from typing import Tuple, Union

import tensorflow as tf

from nebullvm.base import ModelParams
from nebullvm.inference_learners import (
    TensorflowBaseInferenceLearner,
    LearnerMetadata,
)


# TODO: Add TFlite model reading and loading
class TensorflowBackendInferenceLearner(TensorflowBaseInferenceLearner):
    MODEL_NAME = "tf_model.h5"

    def __init__(self, tf_model: tf.Module, **kwargs):
        super(TensorflowBackendInferenceLearner, self).__init__(**kwargs)
        self.model = tf_model

    @tf.function(jit_compile=True)
    def run(self, *input_tensors: tf.Tensor) -> Tuple[tf.Tensor, ...]:
        res = self.model.predict(*input_tensors)
        if not isinstance(res, tuple):
            return (res,)
        return res

    def save(self, path: Union[str, Path], **kwargs):
        path = Path(path)
        metadata = LearnerMetadata.from_model(self, **kwargs)
        metadata.save(path)
        self.model.save(path / self.MODEL_NAME)

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        path = Path(path)
        metadata = LearnerMetadata.read(path)
        network_parameters = ModelParams(**metadata.network_parameters)
        input_tfms = metadata.input_tfms
        model = tf.keras.models.load_model(path / cls.MODEL_NAME)
        return cls(
            tf_model=model,
            network_parameters=network_parameters,
            input_tfms=input_tfms,
        )
