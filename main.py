import torch
import torchvision.models as models
from nebullvm.api.functions import optimize_model

COMPILER_LIST = [
    "deepsparse",
    "tensor RT",
    "torchscript",
    "onnxruntime",
    "tflite",
    "tvm",
    "openvino",
    "bladedisc",
    "intel_neural_compressor",
]

# Load a resnet as example
model = models.resnet50()

# Provide an input data for the model
input_data = [((torch.randn(1, 3, 256, 256),), torch.tensor([0]))]

# Run nebullvm optimization in one line of code
optimized_model = optimize_model(
    model,
    input_data=input_data,
    optimization_time="unconstrained",
    metric_drop_ths=2,
)

# Try the optimized model
x = torch.randn(1, 3, 256, 256)
res = optimized_model(x)
