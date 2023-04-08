from nebullvm.operations.inference_learners.pytorch import (
    PytorchBackendInferenceLearner,
)


class FasterTransformerInferenceLearner(PytorchBackendInferenceLearner):
    MODEL_NAME = "faster_transformer_model_scripted.pt"
    name = "FasterTransformer"
