from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, Union

import tensorflow as tf

from nebullvm.base import DeepLearningFramework, ModelParams
from nebullvm.converters import ONNXConverter
from nebullvm.converters.tensorflow_converters import get_outputs_sizes_tf
from nebullvm.optimizers.multi_compiler import MultiCompilerOptimizer


def optimize_tf_model(
    model: Union[tf.Module, tf.keras.Model],
    batch_size: int,
    input_sizes: List[Tuple[int, ...]],
    save_dir: str,
):
    """Basic function for optimizing a tensorflow model.

    This function saves the output model as well in a nebuly-readable format
    in order to avoid temporary-files corruptions which would prevent the model
    saving later in the process.

    Args:
        model (tf.Module or keras.Model): Model that needs optimization.
        batch_size (int): The model batch size. Note that nebullvm does not
            support at the moment dynamic batch size, so a valid input should
            be given.
        input_sizes (List[Tuple]]): List containing the size of all the input
            tensors of the model. Note that even just a single tensor is needed
            as model input, this field must be a list containing (in the
            exposed case) a single element). The tuple must contain all the
            input tensor dimensions excluding the batch size. This means that
            the final input tensor size will be considered as
            `(batch_size, *input_tensor_size)`, where `input_tensor_size` is
            one list element of `input_sizes`.
        save_dir (str): Path to the directory where saving the final model.

    Returns:
        BaseInferenceLearner: Optimized model usable with the classical
            tensorflow interface. Note that as a torch model it takes as input
            and it gives as output `tf.Tensor`s.
    """
    dl_library = DeepLearningFramework.TENSORFLOW
    model_params = ModelParams(
        batch_size=batch_size,
        input_sizes=input_sizes,
        output_sizes=get_outputs_sizes_tf(
            model,
            input_tensors=[
                tf.random_normal_initializer()(shape=(batch_size, *input_size))
                for input_size in input_sizes
            ],
        ),
    )
    model_converter = ONNXConverter()
    model_optimizer = MultiCompilerOptimizer(n_jobs=-1)
    with TemporaryDirectory() as tmp_dir:
        onnx_path = model_converter.convert(
            model, model_params.input_sizes, Path(tmp_dir)
        )
        model_optimized = model_optimizer.optimize(
            str(onnx_path), dl_library, model_params
        )
        model_optimized.save(save_dir)
    return model_optimized.load(save_dir)
