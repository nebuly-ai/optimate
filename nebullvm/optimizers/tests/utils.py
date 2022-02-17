from nebullvm.base import ModelParams


def get_onnx_model():
    model_path = "test_data/test_model.onnx"
    model_params = {
        "batch_size": 1,
        "input_sizes": [(3, 256, 256)],
        "output_sizes": [(2,)],
    }
    return model_path, ModelParams(**model_params)
