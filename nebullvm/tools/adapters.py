import abc
import copy
from abc import abstractmethod
from typing import List, Any

from nebullvm.core.models import Device, DeviceType
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
from nebullvm.tools.utils import (
    is_huggingface_data,
    check_module_version,
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
        self, inference_learner
    ) -> BaseInferenceLearner:
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

    def adapt_inference_learner(self, model) -> BaseInferenceLearner:
        pipe = copy.deepcopy(self.original_pipeline)
        if self.device.type is DeviceType.GPU:
            pipe.to("cuda")
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        pipe = postprocess_diffusers(
            model,
            pipe,
            self.device,
        )
        return DiffusionInferenceLearner(pipe)


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

    def adapt_inference_learner(self, optimized_model) -> BaseInferenceLearner:
        from nebullvm.operations.inference_learners.huggingface import (
            HuggingFaceInferenceLearner,
        )

        return HuggingFaceInferenceLearner(
            core_inference_learner=optimized_model,
            output_structure=self.__hf_output_structure,
            input_names=self.__hf_input_names,
            output_type=self.__hf_output_type,
        )
