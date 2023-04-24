from abc import ABC
from collections import OrderedDict
from pathlib import Path
from typing import List, Any, Dict, Union

from nebullvm.operations.inference_learners.base import (
    InferenceLearnerWrapper,
    PytorchBaseInferenceLearner,
    LearnerMetadata,
    BaseInferenceLearner,
)
from nebullvm.optional_modules.diffusers import StableDiffusionPipeline
from nebullvm.optional_modules.torch import torch
from nebullvm.tools.diffusers import postprocess_diffusers
from nebullvm.tools.huggingface import restructure_output
from nebullvm.tools.pytorch import get_torch_model_size


class HuggingFaceInferenceLearner(InferenceLearnerWrapper):
    """Class wrapping an InferenceLearner model and giving to it the
    huggingface interface.

    The class fuse both the InterfaceLearner and HuggingFace interfaces, giving
    to the final user a model which can be used whit the prefered API without
    the need of adapting the previous code.

    Attributes:
        network_parameters (ModelParams): Model parameters of the model.
        core_inference_learner (PytorchBaseInferenceLearner): Inference learner
            built using the Pytorch interface.
        output_structure (Dict): Original output structure of the HuggingFace
            model.
        input_names (List[str]): List of all the input keys used for the
            original HuggingFace model.
        output_type (Any, optional): Original output type of the HuggingFace
            model.
    """

    @property
    def name(self) -> str:
        return self.core_inference_learner.name

    def __init__(
        self,
        core_inference_learner: PytorchBaseInferenceLearner,
        output_structure: OrderedDict,
        input_names: List[str],
        output_type: Any = None,
    ):
        super().__init__(core_inference_learner)
        self.output_structure = output_structure
        self.input_names = input_names
        self.output_type = output_type

    def _save_wrapper_extra_info(self):
        pass

    def get_size(self):
        return self.core_inference_learner.get_size()

    @staticmethod
    def _load_wrapper_extra_info(builder_inputs: Dict) -> Dict:
        return builder_inputs

    def run(self, *args, **kwargs) -> Any:
        """Run the underlying optimized model for getting a prediction.

        The method has an hybrid interface. It accepts inputs either as
        positional or keyword arguments. If only positional arguments are given
        the method expects the inputs to be in the canonical
        nebullvm interface. If only keyword arguments are given the method
        expects them to be in the HuggingFace interface. Mixed representation
        is not allowed and will result in an error.
        """
        if len(args) > 0 and len(kwargs) > 0:
            raise RuntimeError(
                "Not allowed usage of the predict method. "
                "Either the positional or the keyword arguments must be given."
            )
        if len(args) > 0:
            return self.core_inference_learner(*args)
        inputs = (kwargs.pop(name) for name in self.input_names)
        outputs = self.core_inference_learner(*inputs)

        if self.output_type is tuple:
            return outputs
        else:
            return restructure_output(
                outputs, self.output_structure, self.output_type
            )

    def _get_extra_metadata_kwargs(self) -> Dict:
        metadata_kwargs = {
            "output_structure": self.output_structure,
            "output_structure_keys": list(self.output_structure.keys()),
            "input_names": self.input_names,
        }
        if self.output_type is not None:
            metadata_kwargs.update(
                {
                    "output_type": self.output_type.__name__,
                    "output_type_module": self.output_type.__module__,
                }
            )
        return metadata_kwargs

    @staticmethod
    def _convert_metadata_to_inputs(metadata: LearnerMetadata) -> Dict:
        # we need to guarantee the preservation of the output structure
        # elements order.
        output_structure = OrderedDict()
        for key in metadata["output_structure_keys"]:
            output_structure[key] = metadata["output_structure"][key]

        inputs = {
            "output_structure": output_structure,
            "input_names": metadata["input_names"],
        }
        if metadata["output_type"] is not None:
            exec(
                f"from {metadata['output_type_module']} "
                f"import {metadata['output_type']}"
            )
            inputs["output_type"] = eval(metadata["output_type"])
        return inputs


class DiffusionInferenceLearner(BaseInferenceLearner, ABC):
    @property
    def name(self) -> str:
        return self.pipeline.unet.model.name

    def __init__(self, pipeline: StableDiffusionPipeline):
        self.pipeline = pipeline

    def __call__(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)

    def run(self, *args, **kwargs) -> Any:
        self.pipeline(*args, **kwargs)

    def save(self, path: Union[str, Path], **kwargs):
        self.pipeline.unet.model.save(path)

    @classmethod
    def load(
        cls,
        path: Union[Path, str],
        **kwargs,
    ):
        try:
            pipe = kwargs["pipe"]
        except KeyError:
            raise TypeError("Missing required argument 'pipe'")
        optimized_model = LearnerMetadata.read(path).load_model(path)
        return postprocess_diffusers(
            optimized_model,
            pipe,
            optimized_model.device,
        )

    def get_size(self):
        (
            self.pipeline.unet.model.get_size()
            + sum(
                [
                    get_torch_model_size(v)
                    for (k, v) in self.pipeline.__dict__.items()
                    if isinstance(v, torch.nn.Module) and k != "unet"
                ]
            )
            / 1e6
        )

    def free_gpu_memory(self):
        raise self.pipeline.unet.model.free_gpu_memory()

    def get_inputs_example(self):
        raise NotImplementedError()

    @property
    def output_format(self):
        return ".pt"

    @property
    def input_format(self):
        return ".pt"

    def list2tensor(self, listified_tensor: List) -> Any:
        raise NotImplementedError()
