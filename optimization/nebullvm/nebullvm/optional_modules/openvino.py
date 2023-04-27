import logging

from nebullvm.optional_modules.dummy import DummyClass

try:
    from openvino.runtime import Core, Model, CompiledModel, InferRequest
    from openvino.tools.pot import DataLoader
    from openvino.tools.pot import IEEngine
    from openvino.tools.pot import load_model, save_model
    from openvino.tools.pot import compress_model_weights
    from openvino.tools.pot import create_pipeline
except ImportError:
    Model = CompiledModel = InferRequest = Core = DummyClass
    DataLoader = IEEngine = DummyClass
    load_model = save_model = compress_model_weights = create_pipeline = None

# Fix openvino issue with logging
# It adds a second handler to the root logger that cause issues
if len(logging.getLogger().handlers) > 1:
    logging.getLogger().removeHandler(logging.getLogger().handlers[-1])
