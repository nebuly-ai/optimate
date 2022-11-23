import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50

# Load a resnet as example
from nebullvm.apps import BlackBoxModelOptimization

model = ResNet50()

# Provide an input data for the model
input_data = [((tf.random.normal([1, 224, 224, 3]),), tf.constant([0]))]

model_optim = BlackBoxModelOptimization()
model_optim.execute(
    model,
    input_data,
    metric_drop_ths=2,
)
optimized_model = model_optim.root_op.optimal_model
