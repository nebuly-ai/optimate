import abc
from typing import Any, Dict, List, Optional

from nebullvm.operations.base import Operation
from nebullvm.tools.base import QuantizationType


class Compiler(Operation, abc.ABC):
    supported_ops: Dict[str, List[Optional[QuantizationType]]]

    def __init__(self):
        super().__init__()
        self.compiled_model = None

    @abc.abstractmethod
    def execute(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _compile_model(self, **kwargs) -> Any:
        raise NotImplementedError()

    @abc.abstractmethod
    def _quantize_model(self, **kwargs) -> Any:
        raise NotImplementedError()

    def get_result(self) -> Any:
        return self.compiled_model
