try:
    import diffusers  # noqa F401
    from diffusers import (
        StableDiffusionPipeline,
        DiffusionPipeline,
    )  # noqa F401
    from diffusers.models import (
        AutoencoderKL,
        UNet2DConditionModel,
    )  # noqa F401
    from diffusers.models.unet_2d import UNet2DOutput  # noqa F401
    import onnx_graphsurgeon  # noqa F401
except ImportError:
    diffusers = None
    StableDiffusionPipeline = None
    DiffusionPipeline = None
    UNet2DConditionModel = None
    AutoencoderKL = None
    UNet2DOutput = None
    onnx_graphsurgeon = None
