from typing import List, Tuple, Any

import numpy as np

from nebullvm.config import NO_COMPILER_INSTALLATION

try:
    from openvino.tools.pot import DataLoader
    from openvino.tools.pot import IEEngine
    from openvino.tools.pot import load_model, save_model
    from openvino.tools.pot import compress_model_weights
    from openvino.tools.pot import create_pipeline
except ImportError:
    import cpuinfo

    if (
        "intel" in cpuinfo.get_cpu_info()["brand_raw"].lower()
        and not NO_COMPILER_INSTALLATION
    ):
        from nebullvm.installers.installers import install_openvino

        install_openvino()
        from openvino.tools.pot import DataLoader
        from openvino.tools.pot import IEEngine
        from openvino.tools.pot import load_model, save_model
        from openvino.tools.pot import compress_model_weights
        from openvino.tools.pot import create_pipeline
    else:
        DataLoader = object


class _CalibrationDataLoader(DataLoader):
    def __init__(
        self, input_data: List[Tuple[Any, ...]], input_names: List[str]
    ):
        self._input_data = input_data
        self._input_names = input_names

    def __len__(self):
        return len(self._input_data[0])

    def __getitem__(self, item):
        return (
            dict(zip(self._input_names, self._input_data[0][item])),
            self._input_data[1][item],
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
