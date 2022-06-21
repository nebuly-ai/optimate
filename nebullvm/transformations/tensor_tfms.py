from typing import Any

import torch

from nebullvm.transformations.base import BaseTransformation


class VerifyContiguity(BaseTransformation):
    def _transform(self, _input: torch.Tensor, **kwargs) -> Any:
        if not _input.is_contiguous():
            _input = _input.contiguous()
        return _input
