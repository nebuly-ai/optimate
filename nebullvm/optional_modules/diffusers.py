from nebullvm.optional_modules.dummy import DummyClass

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
except ImportError:
    diffusers = DummyClass
    StableDiffusionPipeline = DummyClass
    DiffusionPipeline = DummyClass
    UNet2DConditionModel = DummyClass
    AutoencoderKL = DummyClass
    UNet2DOutput = DummyClass

try:
    import onnx_graphsurgeon  # noqa F401
except ImportError:
    onnx_graphsurgeon = DummyClass
