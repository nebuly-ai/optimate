import logging

logger = logging.getLogger(__name__)

try:
    from openvino.runtime import Core, Model, CompiledModel, InferRequest
    from openvino.tools.pot import DataLoader
    from openvino.tools.pot import IEEngine
    from openvino.tools.pot import load_model, save_model
    from openvino.tools.pot import compress_model_weights
    from openvino.tools.pot import create_pipeline
except ImportError:
    logger.warning(
        "Missing Library: "
        "openvino module is not installed on this platform. "
        "Please install it if you want to include it in the "
        "optimization pipeline."
    )
    Model = CompiledModel = InferRequest = Core = object
    DataLoader = IEEngine = object
    load_model = save_model = compress_model_weights = create_pipeline = None
