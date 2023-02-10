import time
from abc import abstractmethod, ABC
from typing import Any, Dict, Type

import numpy as np
from loguru import logger
from tqdm import tqdm

from nebullvm.operations.inference_learners.base import BaseInferenceLearner
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch, DataLoader
from nebullvm.tools.base import (
    DeepLearningFramework,
    ModelParams,
    DeviceType,
)
from nebullvm.tools.data import DataManager
from nebullvm.tools.onnx import create_model_inputs_onnx
from nebullvm.tools.pytorch import create_model_inputs_torch
from nebullvm.tools.tf import create_model_inputs_tf
from nebullvm.tools.utils import (
    check_input_data,
    extract_info_from_data,
    is_data_subscriptable,
    check_device,
)


def _get_dl_framework(model: Any):
    if isinstance(model, torch.nn.Module) or str(model).startswith("Pytorch"):
        return DeepLearningFramework.PYTORCH
    elif (isinstance(model, tf.Module) and model is not None) or str(
        model
    ).startswith("Tensorflow"):
        return DeepLearningFramework.TENSORFLOW
    elif isinstance(model, str) or str(model).startswith("Numpy"):
        return DeepLearningFramework.NUMPY
    else:
        raise TypeError(f"Model type {type(model)} not supported.")


def _create_model_inputs(
    dl_framework: DeepLearningFramework, model_params: ModelParams
):
    if dl_framework == DeepLearningFramework.PYTORCH:
        input_data = create_model_inputs_torch(model_params.input_infos)
    elif dl_framework == DeepLearningFramework.TENSORFLOW:
        input_data = create_model_inputs_tf(model_params.input_infos)
    elif dl_framework == DeepLearningFramework.NUMPY:
        input_data = create_model_inputs_onnx(model_params.input_infos)
    else:
        raise TypeError(f"Unknown framework {dl_framework}")

    return input_data


class BaseBenchmark(ABC):
    def __init__(self, model, input_tensors, device, n_warmup=50, n_runs=1000):
        self.model = model
        self.input_tensors = input_tensors
        self.device = device
        self.n_warmup = n_warmup
        self.n_runs = n_runs

    @abstractmethod
    def benchmark(self):
        raise NotImplementedError


class PytorchBenchmark(BaseBenchmark):
    def benchmark(self):
        input_tensors = [
            [tensor.to(self.device.to_torch_format()) for tensor in tensors]
            for tensors in self.input_tensors
        ]
        batch_size = input_tensors[0][0].shape[0]

        if isinstance(self.model, torch.nn.Module):
            self.model.to(self.device.to_torch_format()).eval()

        with torch.no_grad():
            for i in tqdm(
                range(self.n_warmup),
                desc=f"Performing warm up on {self.n_warmup} iterations",
            ):
                self.model(
                    *input_tensors[i % min(self.n_warmup, len(input_tensors))]
                )
        if self.device.type is DeviceType.GPU:
            torch.cuda.synchronize()
        timings = []
        with torch.no_grad():
            for i in tqdm(
                range(1, self.n_runs + 1),
                desc=f"Performing benchmark on {self.n_runs} iterations",
            ):
                start_time = time.time()
                self.model(
                    *input_tensors[i % min(self.n_runs, len(input_tensors))]
                )
                if self.device.type is DeviceType.GPU:
                    torch.cuda.synchronize()
                end_time = time.time()
                timings.append(end_time - start_time)

        print(f"Batch size: {batch_size}")

        throughput = batch_size / np.mean(timings)
        latency = np.mean(timings) / batch_size

        print("Average Throughput: %.2f data/second" % throughput)
        print("Average Latency: %.4f seconds/data" % latency)

        return throughput, latency


class TensorflowBenchmark(BaseBenchmark):
    def benchmark(self):
        batch_size = self.input_tensors[0][0].shape[0]

        for i in tqdm(
            range(self.n_warmup),
            desc=f"Performing warm up on {self.n_warmup} iterations",
        ):
            with tf.device(self.device.to_tf_format()):
                self.model(
                    *self.input_tensors[
                        i % min(self.n_warmup, len(self.input_tensors))
                    ]
                )

        timings = []
        for i in tqdm(
            range(1, self.n_runs + 1),
            desc=f"Performing benchmark on {self.n_runs} iterations",
        ):
            start_time = time.time()
            with tf.device(self.device.to_tf_format()):
                self.model(
                    *self.input_tensors[
                        i % min(self.n_runs, len(self.input_tensors))
                    ]
                )

            end_time = time.time()
            timings.append(end_time - start_time)

        print(f"Batch size: {batch_size}")

        throughput = batch_size / np.mean(timings)
        latency = np.mean(timings) / batch_size

        print("Average Throughput: %.2f data/second" % throughput)
        print("Average Latency: %.4f seconds/data" % latency)

        return throughput, latency


class NumpyBenchmark(BaseBenchmark):
    def benchmark(self):
        if not isinstance(self.model, BaseInferenceLearner):
            # TODO: Add support for original onnx models
            raise NotImplementedError(
                "Benchmark function doesn't support original " "onnx models."
            )
        batch_size = self.input_tensors[0][0].shape[0]

        for i in tqdm(
            range(self.n_warmup),
            desc=f"Performing warm up on {self.n_warmup} iterations",
        ):
            self.model(
                *self.input_tensors[
                    i % min(self.n_warmup, len(self.input_tensors))
                ]
            )

        timings = []
        for i in tqdm(
            range(1, self.n_runs + 1),
            desc=f"Performing benchmark on {self.n_runs} iterations",
        ):
            start_time = time.time()
            self.model(
                *self.input_tensors[
                    i % min(self.n_runs, len(self.input_tensors))
                ]
            )

            end_time = time.time()
            timings.append(end_time - start_time)

        print(f"Batch size: {batch_size}")

        throughput = batch_size / np.mean(timings)
        latency = np.mean(timings) / batch_size

        print("Average Throughput: %.2f data/second" % throughput)
        print("Average Latency: %.4f seconds/data" % latency)

        return throughput, latency


def benchmark(
    model, input_data, device=None, random=False, n_warmup=50, n_runs=1000
):
    """Performs a Benchmark on the input model regardless of the framework it
    was used for implementing it.
    Args:
        model (Any): The input model.
        input_data (Iterable or Sequence): Input data to be used for
            optimizing the model. PyTorch, TensorFlow
            and Onnx respectively accept input tensor in `torch.Tensor`,
            `tf.Tensor` and `np.ndarray` formats. Note that the each input
            sample must be a tuple containing a tuple as first element, the
            `inputs`, and the `label` as second element. The `inputs` needs to
            be passed as tuple even if a single input is needed by the model
            (in this case the `inputs` tuple will contain just an element).
            HuggingFace models can take as data samples both dictionaries or
            strings. Strings will then be converted in data samples using the
            HuggingFace tokenizer which must be given as input when just a
            list of string is provided as input_data (tokenizers can be passed
            as extra arguments of this function using the keyword `tokenizer`).
        device (str): Device to be used for running the benchmark. If None,
            CPU will be used. Default: None.
        random (bool, optional): If set to true, the data used to benchmark the
            model will be computed randomly given the info extracted from the
            provided input_data.
        n_warmup (int, optional): Number of warmup iterations.
        n_runs (int, optional): Number of iterations performed to benchmark
            the model.
    """
    if not isinstance(model, BaseInferenceLearner):
        device = check_device(device)
    else:
        device = model.device

    logger.info(f"Running benchmark on {device.type.name}")

    dl_framework = _get_dl_framework(model)

    if isinstance(input_data, (DataLoader, tf.data.Dataset)):
        try:
            input_data = DataManager.from_dataloader(input_data)
        except Exception:
            raise ValueError(
                "The provided dataloader does not match the expected "
                "format.\n"
                "Speedster supports dataloaders that return tuples in "
                "the\n"
                "following formats: \n"
                "Single input: (input,  label)\n"
                "Multiple inputs: ((input1, input2, ...),  label) or "
                "(input1, input2, ...,  label)\n"
                "Inputs and labels should be either tensors or numpy "
                "arrays,\n"
                "depending on the framework used.\n"
            )

    if not isinstance(input_data, DataManager):
        if check_input_data(input_data):
            if is_data_subscriptable(input_data):
                input_data = DataManager(input_data)
            else:
                input_data = DataManager.from_iterable(input_data)
        else:
            raise ValueError(
                "The provided data does not match the expected "
                "format.\n"
                "Speedster supports data in the following formats: \n"
                "- PyTorch DataLoader\n"
                "- TensorFlow Dataset\n"
                "- List of tuples: [((input_0, ... ), label), ...] \n"
                "Inputs and labels should be either tensors or numpy "
                "arrays,\n"
                "depending on the framework used.\n"
            )

    if random:
        model_params = extract_info_from_data(
            model, input_data, dl_framework, None, device
        )
        input_data = _create_model_inputs(dl_framework, model_params)
    else:
        input_data = input_data.get_list()

    BENCHMARK_FUNCTIONS[dl_framework](
        model=model,
        input_tensors=input_data,
        device=device,
        n_warmup=n_warmup,
        n_runs=n_runs,
    ).benchmark()


BENCHMARK_FUNCTIONS: Dict[DeepLearningFramework, Type[BaseBenchmark]] = {
    DeepLearningFramework.PYTORCH: PytorchBenchmark,
    DeepLearningFramework.TENSORFLOW: TensorflowBenchmark,
    DeepLearningFramework.NUMPY: NumpyBenchmark,
}
