import os

LEARNER_METADATA_FILENAME = "metadata.json"
NO_COMPILER_INSTALLATION = int(os.getenv("NO_COMPILER_INSTALLATION", "0")) > 0

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

OPENVINO_FILENAMES = {
    "metadata": LEARNER_METADATA_FILENAME,
    "description_file": "description.xml",
    "weights": "weights.bin",
}
