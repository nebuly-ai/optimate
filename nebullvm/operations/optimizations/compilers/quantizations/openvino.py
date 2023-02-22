from typing import List, Tuple, Any

import numpy as np

from nebullvm.optional_modules.openvino import (
    DataLoader,
    load_model,
    IEEngine,
    create_pipeline,
    compress_model_weights,
    save_model,
)


class _CalibrationDataLoader(DataLoader):
    def __init__(
        self, input_data: List[Tuple[Any, ...]], input_names: List[str]
    ):
        self._input_data = input_data
        self._input_names = input_names

    def __len__(self):
        return len(self._input_data)

    def __getitem__(self, item):
        inputs = {
            k: v for (k, v) in zip(self._input_names, self._input_data[item])
        }
        return (
            (item, None),
            inputs,
        )


def quantize_openvino(
    model_topology: str,
    model_weights: str,
    input_data: List[Tuple[np.ndarray, ...]],
    input_names: List[str],
) -> Tuple[str, str]:
    model_config = {
        "model_name": "model",
        "model": model_topology,
        "weights": model_weights,
    }

    # Engine config
    engine_config = {"device": "CPU"}

    algorithms = [
        {
            "name": "DefaultQuantization",
            "params": {
                "target_device": "ANY",
                "preset": "performance",
                "stat_subset_size": len(input_data),
            },
        }
    ]
    data_loader = _CalibrationDataLoader(
        input_data=input_data, input_names=input_names
    )
    model = load_model(model_config=model_config)
    engine = IEEngine(config=engine_config, data_loader=data_loader)
    pipeline = create_pipeline(algorithms, engine)
    compressed_model = pipeline.run(model=model)
    compress_model_weights(compressed_model)
    compressed_model_paths = save_model(
        model=compressed_model,
        save_path="quantized_model",
        model_name="quantized_model",
    )

    return (
        compressed_model_paths[0]["model"],
        compressed_model_paths[0]["weights"],
    )
