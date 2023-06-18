from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Callable, Tuple

import yaml

from nebullvm.operations.base import Operation
from nebullvm.tools.data import DataManager


class Compressor(Operation, ABC):
    def __init__(self, config_file: str = None):
        super().__init__()
        self._config = self._read_config(config_file)
        self.compressed_model = None
        self.new_metric_ths = None

    @abstractmethod
    def execute(
        self,
        model: Any,
        train_input_data: DataManager,
        eval_input_data: DataManager,
        metric_drop_ths: float,
        metric: Callable,
    ) -> Tuple[Any, Optional[float]]:
        raise NotImplementedError()

    def _read_config(self, config_file: Optional[str]) -> Dict:
        config = self._get_default_config()
        if config_file is not None:
            with open(config_file, "r") as f:
                data = yaml.load(f, Loader=yaml.CLoader)
                config.update(data.get(self.config_key, {}))
        return config

    @staticmethod
    @abstractmethod
    def _get_default_config() -> Dict:
        raise NotImplementedError

    @property
    @abstractmethod
    def config_key(self) -> str:
        raise NotImplementedError()

    def get_result(self) -> Tuple[Any, Optional[float]]:
        return self.compressed_model, self.new_metric_ths
