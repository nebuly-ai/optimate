import cpuinfo
import warnings
from abc import ABC
from pathlib import Path
from typing import Union, Tuple, Dict, Type

import torch.fx

from nebullvm.base import DeepLearningFramework, ModelParams
from nebullvm.config import NO_COMPILER_INSTALLATION
from nebullvm.inference_learners.base import (
    BaseInferenceLearner,
    LearnerMetadata,
    PytorchBaseInferenceLearner,
)
from nebullvm.installers.installers import install_intel_neural_compressor
from nebullvm.transformations.base import MultiStageTransformation

try:
    from neural_compressor.utils.pytorch import load
except ImportError:
    import platform

    os_ = platform.system()
    if (
        "intel" in cpuinfo.get_cpu_info()["brand_raw"].lower()
        and not NO_COMPILER_INSTALLATION
    ):
        warnings.warn(
            "No intel neural compressor installation found. "
            "Trying to install it..."
        )
        install_intel_neural_compressor()
        try:
            from neural_compressor.utils.pytorch import load
        except ImportError:
            # Solves a problem in colab
            pass
    else:
        warnings.warn(
            "No valid intel neural compressor installation found. "
            "The compiler won't be used in the following."
        )


class NeuralCompressorInferenceLearner(BaseInferenceLearner, ABC):
    """Model optimized on CPU using IntelNeuralCompressor.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        model (torch.fx.GraphModule): Torch fx graph model.
    """

    def __init__(
        self,
        model: torch.fx.GraphModule,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model

    def save(self, path: Union[str, Path], **kwargs):
        """Save the model.

        Args:
            path (Path or str): Path to the directory where the model will
                be stored.
            kwargs (Dict): Dictionary of key-value pairs that will be saved in
                the model metadata file.
        """
        metadata = LearnerMetadata.from_model(self, **kwargs)
        metadata.save(path)

        self.model.save(str(path))

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs):
        """Load the model.

        Args:
            path (Path or str): Path to the directory where the model is
                stored.
            kwargs (Dict): Dictionary of additional arguments for consistency
                with other Learners.

        Returns:
            DeepSparseInferenceLearner: The optimized model.
        """
        if len(kwargs) > 0:
            warnings.warn(
                f"No extra keywords expected for the load method. "
                f"Got {kwargs}."
            )
        model = load("./output")
        metadata = LearnerMetadata.read(path)
        input_tfms = metadata.input_tfms
        if input_tfms is not None:
            input_tfms = MultiStageTransformation.from_dict(
                metadata.input_tfms
            )
        return cls(
            input_tfms=input_tfms,
            network_parameters=ModelParams(**metadata.network_parameters),
            model=model,
        )


class PytorchNeuralCompressorInferenceLearner(
    NeuralCompressorInferenceLearner, PytorchBaseInferenceLearner
):
    """Model optimized on CPU using IntelNeuralCompressor.

    Attributes:
        network_parameters (ModelParams): The model parameters as batch
                size, input and output sizes.
        model (torch.fx.GraphModule): Torch fx graph model.
    """

    def run(self, *input_tensors: torch.Tensor) -> Tuple[torch.Tensor]:
        """Predict on the input tensors.

        Note that the input tensors must be on the same batch. If a sequence
        of tensors is given when the model is expecting a single input tensor
        (with batch size >= 1) an error is raised.

        Args:
            input_tensors (Tuple[Tensor]): Input tensors belonging to the same
                batch. The tensors are expected having dimensions
                (batch_size, dim1, dim2, ...).

        Returns:
            Tuple[Tensor]: Output tensors. Note that the output tensors does
                not correspond to the prediction on the input tensors with a
                1 to 1 mapping. In fact the output tensors are produced as the
                multiple-output of the model given a (multi-) tensor input.
        """
        outputs = self.model(*input_tensors)
        return outputs


NEURAL_COMPRESSOR_INFERENCE_LEARNERS: Dict[
    DeepLearningFramework, Type[NeuralCompressorInferenceLearner]
] = {DeepLearningFramework.PYTORCH: PytorchNeuralCompressorInferenceLearner}
