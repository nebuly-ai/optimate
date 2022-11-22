import torch
import torchvision.models as models
from nebullvm.apps import BlackBoxModelOptimization

# Load a resnet as example
model = models.resnet50()

# Provide an input data for the model
input_data = [((torch.randn(1, 3, 256, 256),), torch.tensor([0]))]

dynamic_info = {"inputs": [{0: "batch"}], "outputs": [{0: "batch"}]}

model_optim = BlackBoxModelOptimization()
model_optim.execute(
    model,
    input_data,
    metric_drop_ths=2,
    dynamic_info=dynamic_info,
)
optimized_model = model_optim.root_op.optimal_model
