import torch
import torchvision.models as models
from nebullvm.api.functions import optimize_model

# Load a resnet as example
model = models.resnet50()

# Provide an input data for the model
input_data = [
    ((torch.randn(1, 3, 256, 256),), torch.tensor([0])) for i in range(100)
]

# Run nebullvm optimization in one line of code
optimized_model = optimize_model(
    model,
    input_data=input_data,
    optimization_time="constrained",
    metric_drop_ths=0.2,
)

# Try the optimized model
x = torch.randn(1, 3, 256, 256)
res = optimized_model(x)
