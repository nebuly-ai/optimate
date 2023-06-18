from nebullvm.operations.inference_learners.torchscript import (
    TorchScriptInferenceLearner,
)


class FasterTransformerInferenceLearner(TorchScriptInferenceLearner):
    MODEL_NAME = "faster_transformer_model_scripted.pt"
    name = "FasterTransformer"
