from pathlib import Path
from typing import Union, Any

from nebullvm.operations.inference_learners.base import LearnerMetadata
from nebullvm.optional_modules.diffusers import StableDiffusionPipeline
from nebullvm.tools.diffusers import postprocess_diffusers


def load_model(path: Union[Path, str], pipe: StableDiffusionPipeline = None):
    """Load the optimized model previously saved in the given path.

    Args:
        path (Union[Path, str]): Path to the directory where the model is
            saved.
        pipe (StableDiffusionPipeline): Diffusion pipeline to be used for
            loading the model. This parameter is only needed if the model
            to be loaded is a diffusion model. Default: None.

    Returns:
        InferenceLearner: Model optimized by Speedster.
    """
    optimized_model = LearnerMetadata.read(path).load_model(path)
    if pipe is not None:
        optimized_model = postprocess_diffusers(
            optimized_model, pipe, optimized_model.device
        )
    return optimized_model


def save_model(model: Any, path: Union[Path, str]):
    """Save the optimized model in the given path.

    Args:
        model (Any): Model to be saved.
        path (Union[Path, str]): Path to the directory where to
            save the model.

    Returns:
        InferenceLearner: Model optimized by Speedster.
    """
    if isinstance(model, StableDiffusionPipeline):
        model.unet.model.save(path)
    else:
        model.save(path)
