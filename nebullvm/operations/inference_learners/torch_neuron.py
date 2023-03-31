from nebullvm.operations.inference_learners.pytorch import (
    PytorchBackendInferenceLearner,
)


class TorchNeuronInferenceLearner(PytorchBackendInferenceLearner):
    name = "TorchNeuron"
