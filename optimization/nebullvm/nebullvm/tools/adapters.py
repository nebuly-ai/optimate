import abc
import copy
from abc import abstractmethod
import time
from typing import List, Any, Union

from loguru import logger

from nebullvm.core.models import (
    Device,
    DeviceType,
    OptimizedModel,
    OriginalModel,
)
from nebullvm.operations.conversions.huggingface import convert_hf_model
from nebullvm.operations.inference_learners.base import (
    BaseInferenceLearner,
)
from nebullvm.operations.inference_learners.huggingface import (
    DiffusionInferenceLearner,
)
from nebullvm.optional_modules.diffusers import StableDiffusionPipeline
from nebullvm.optional_modules.torch import torch
from nebullvm.tools.diffusers import (
    get_unet_inputs,
    preprocess_diffusers,
    postprocess_diffusers,
)
from nebullvm.tools.pytorch import get_torch_model_size
from nebullvm.tools.utils import (
    is_huggingface_data,
    check_module_version,
    get_throughput,
)


class ModelAdapter(abc.ABC):
    @property
    @abstractmethod
    def adapted_model(self):
        pass

    @property
    @abstractmethod
    def adapted_data(self):
        pass

    @abstractmethod
    def adapt_inference_learner(
        self, optimized_model: OptimizedModel
    ) -> BaseInferenceLearner:
        pass

    @abstractmethod
    def adapt_original_model(
        self, original_model: OriginalModel
    ) -> OriginalModel:
        pass


class DiffusionAdapter(ModelAdapter):
    def __init__(
        self,
        original_pipeline: StableDiffusionPipeline,
        data: List,
        device: Device,
    ):
        self.original_pipeline = copy.deepcopy(original_pipeline)
        self.original_data = data
        self.device = device
        self.__adapted = False
        self.__df_model = None
        self.__df_data = None

    @torch.no_grad()
    def __benchmark_pipeline(
        self,
        pipe: Union[StableDiffusionPipeline, BaseInferenceLearner],
        num_warmup_steps=2,
        num_steps=3,
    ):

        # Warmup
        for i in range(num_warmup_steps):
            _ = pipe(self.original_data[i % len(self.original_data)]).images[0]

        start = time.time()
        # Benchmark
        for i in range(num_steps):
            _ = pipe(self.original_data[i % len(self.original_data)]).images[0]

        took = time.time() - start

        return took / num_steps

    def __adapt(self):
        if not check_module_version(torch, max_version="1.13.1+cu117"):
            raise ValueError(
                "Diffusion models are only supported in PyTorch "
                "versions <= 1.13.1. Please downgrade your PyTorch "
                "version and try again."
            )

        model = copy.deepcopy(self.original_pipeline)
        model.get_unet_inputs = get_unet_inputs
        model.to(self.device.to_torch_format())
        self.__df_data = [
            (
                tuple(
                    d.reshape((1,)) if d.shape == torch.Size([]) else d
                    for d in model.get_unet_inputs(
                        model,
                        prompt=prompt,
                    )
                    if d is not None
                ),
                None,
            )
            for prompt in self.original_data
        ]
        self.__df_model = preprocess_diffusers(model)
        self.__adapted = True

    @property
    def adapted_model(self):
        if self.__adapted is False:
            self.__adapt()
        return self.__df_model

    @property
    def adapted_data(self):
        if self.__adapted is False:
            self.__adapt()
        return self.__df_data

    def adapt_inference_learner(
        self, optimized_model: OptimizedModel
    ) -> OptimizedModel:
        pipe = copy.deepcopy(self.original_pipeline)
        pipe.to(self.device.to_torch_format())
        if self.device.type is DeviceType.GPU:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        pipe = postprocess_diffusers(
            optimized_model.inference_learner,
            pipe,
            self.device,
        )
        logger.info("Benchmarking optimized pipeline...")
        optimized_model.latency_seconds = self.__benchmark_pipeline(pipe)
        optimized_model.throughput = get_throughput(
            optimized_model.latency_seconds
        )
        optimized_model.inference_learner = DiffusionInferenceLearner(pipe)
        optimized_model.size_mb += (
            sum(
                [
                    get_torch_model_size(v)
                    for (k, v) in pipe.__dict__.items()
                    if isinstance(v, torch.nn.Module) and k != "unet"
                ]
            )
            / 1e6
        )
        return optimized_model

    def adapt_original_model(
        self, original_model: OriginalModel
    ) -> OriginalModel:
        pipe = copy.deepcopy(self.original_pipeline)
        pipe.to(self.device.to_torch_format())
        logger.info("Benchmarking original pipeline...")
        original_model.latency_seconds = self.__benchmark_pipeline(pipe)
        original_model.throughput = get_throughput(
            original_model.latency_seconds
        )
        original_model.size_mb += (
            sum(
                [
                    get_torch_model_size(v)
                    for (k, v) in pipe.__dict__.items()
                    if isinstance(v, torch.nn.Module) and k != "unet"
                ]
            )
            / 1e6
        )
        return original_model


class HuggingFaceAdapter(ModelAdapter):
    def __init__(self, model: Any, data: List, device: Device, **kwargs):
        self.original_model = model
        self.original_data = data
        self.device = device
        self.tokenizer_params = kwargs
        self.__adapted = False
        self.__hf_model = None
        self.__hf_data = None
        self.__hf_input_names = None
        self.__hf_output_type = None
        self.__hf_output_structure = None

    def __adapt_model(self):
        if not is_huggingface_data(self.original_data[0]):
            raise ValueError("Cannot convert non-HuggingFace data")
        (
            model,
            data,
            input_names,
            output_structure,
            output_type,
        ) = convert_hf_model(
            self.original_model,
            self.original_data,
            self.device,
            **self.tokenizer_params,
        )
        self.__hf_model = model
        self.__hf_data = data
        self.__hf_input_names = input_names
        self.__hf_output_type = output_type
        self.__hf_output_structure = output_structure
        self.__adapted = True

    @property
    def adapted_model(self):
        if self.__adapted is False:
            self.__adapt_model()
        return self.__hf_model

    @property
    def adapted_data(self):
        if self.__adapted is False:
            self.__adapt_model()
        return self.__hf_data

    def adapt_inference_learner(
        self, optimized_model: OptimizedModel
    ) -> OptimizedModel:
        from nebullvm.operations.inference_learners.huggingface import (
            HuggingFaceInferenceLearner,
        )

        optimized_model.inference_learner = HuggingFaceInferenceLearner(
            core_inference_learner=optimized_model.inference_learner,
            output_structure=self.__hf_output_structure,
            input_names=self.__hf_input_names,
            output_type=self.__hf_output_type,
        )

        return optimized_model

    def adapt_original_model(
        self, original_model: OriginalModel
    ) -> OriginalModel:
        return original_model
