from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Callable, Tuple

import yaml

from nebullvm.utils.data import DataManager


class BaseCompressor(ABC):
    def __init__(self, config_file: str = None):
        self._config = self._read_config(config_file)

    @abstractmethod
    def compress(
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
