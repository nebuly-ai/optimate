try:
    import diffusers  # noqa F401
    from diffusers.models import UNet2DConditionModel  # noqa F401
except ImportError:
    diffusers = None
