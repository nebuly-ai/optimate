from nebullvm.optional_modules.torch import torch


VERSION = "0.8.1"
LEARNER_METADATA_FILENAME = "metadata.json"
ONNX_OPSET_VERSION = 13
NEBULLVM_DEBUG_FILE = "nebullvm_debug.json"

AUTO_TVM_TUNING_OPTION = {
    "tuner": "xgb",
    "trials": 10,
    "early_stopping": 100,
}
# TODO: remove the min_repeat_ms key
AUTO_TVM_PARAMS = {
    "number": 10,
    "repeat": 1,
    "min_repeat_ms": 0,  # since we're tuning on a CPU, can be set to 0
    "timeout": 10,  # in seconds
}

NVIDIA_FILENAMES = {
    "engine": "tensor_rt.engine",
    "metadata": LEARNER_METADATA_FILENAME,
}

TVM_FILENAMES = {"engine": "compiled_lib.so"}

ONNX_FILENAMES = {"model_name": "model.onnx"}
ONNX_PROVIDERS = {
    "cuda": [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
    "cpu": [
        "CPUExecutionProvider",
    ],
}

OPENVINO_FILENAMES = {
    "metadata": LEARNER_METADATA_FILENAME,
    "description_file": "description.xml",
    "weights": "weights.bin",
}

TENSORFLOW_BACKEND_FILENAMES = {
    "tflite_model": "tf_model.tflite",
    "tf_model": "tf_model.h5",
}

TORCH_TENSORRT_PRECISIONS = {
    "torch.float32": {torch.float},
    "torch.float16": {torch.float, torch.half},
    "torch.int8": {torch.float, torch.half, torch.int8},
}

MIN_DIM_INPUT_DATA = 100
QUANTIZATION_DATA_NUM = 300
CONSTRAINED_METRIC_DROP_THS = 1e-2
TRAIN_TEST_SPLIT_RATIO = 0.8

COMPILER_LIST = [
    "deepsparse",
    "tensor_rt",
    "torchscript",
    "onnxruntime",
    "tflite",
    "xla",
    "tvm",
    "openvino",
    "bladedisc",
    "intel_neural_compressor",
]

COMPRESSOR_LIST = [
    "sparseml",
    "intel_pruning",
]

ONNX_MODULES = ["openvino", "tensor_rt"]

TORCH_MODULES = [
    "deepsparse",
    "intel_neural_compressor",
    "tensor_rt",
    "torch_tensor_rt",
]

TENSORFLOW_MODULES = []
HUGGING_FACE_MODULES = []

LIBRARIES_GPU = ["tensor_rt", "torch_tensor_rt"]

MIN_NUMBER = 1e-4
DEFAULT_METRIC_DROP_THS = 1e-3
ACTIVATION_METRIC_DROP_THS = 1e-2
