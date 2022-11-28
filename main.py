import torch

# import torchvision.models as models
from nebullvm import optimize_model

# Load a resnet as example
# model = models.alexnet()
model = torch.nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)

# Provide an input data for the model
input_data = [
    ((torch.randn(1, 3, 224, 224),), torch.tensor([0])) for i in range(100)
]

dynamic_info = {"inputs": [{0: "batch"}], "outputs": [{0: "batch"}]}

model_optim = optimize_model(
    model,
    input_data,
    metric_drop_ths=2,
    dynamic_info=dynamic_info,
    optimization_time="unconstrained",
    store_latencies=True,
    ignore_compilers=["torchscript", "onnxruntime"],
)

print(model_optim)
