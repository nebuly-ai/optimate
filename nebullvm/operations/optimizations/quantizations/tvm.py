from typing import List, Sequence, Any, Dict

from nebullvm.config import QUANTIZATION_DATA_NUM
from nebullvm.operations.optimizations.quantizations.base import Quantizer
from nebullvm.optional_modules.tvm import (
    IRModule,
    NDArray,
    relay,
    ToMixedPrecision,
)
from nebullvm.tools.base import QuantizationType
from nebullvm.tools.data import DataManager
from nebullvm.tools.transformations import (
    MultiStageTransformation,
    HalfPrecisionTransformation,
)


class TVMCalibrator(DataManager):
    def __init__(self, data_reader: Sequence, input_names: List[str]):
        super(TVMCalibrator, self).__init__(data_reader=data_reader)
        self._input_names = input_names

    def __getitem__(self, item: int):
        tuple_ = self._data_reader[item]
        return {name: data for name, data in zip(self._input_names, tuple_)}


class ApacheTVMQuantizer(Quantizer):
    def execute(
        self,
        model: Any,
        quantization_type: QuantizationType,
        input_tfms: MultiStageTransformation,
        input_data: DataManager,
        params,
    ):
        if quantization_type is not None:
            if quantization_type is QuantizationType.HALF:
                self.quantized_model = ToMixedPrecision(
                    mixed_precision_type="float16"
                )(model)
                input_tfms.append(HalfPrecisionTransformation())
            else:
                if quantization_type is QuantizationType.DYNAMIC:
                    inputs = None
                elif quantization_type is QuantizationType.STATIC:
                    inputs = input_data.get_split("train").get_numpy_list(
                        QUANTIZATION_DATA_NUM
                    )
                    input_names = [f"input_{n}" for n in range(len(inputs[0]))]
                    inputs = TVMCalibrator(inputs, input_names)
                else:
                    return
                self.quantized_model = self._quantize(
                    model, params, input_data=inputs
                )

    @staticmethod
    def _quantize(
        mod: IRModule,
        params: Dict[str, NDArray],
        input_data: TVMCalibrator = None,
    ) -> IRModule:
        if input_data is not None:
            with relay.quantize.qconfig(
                calibrate_mode="kl_divergence", weight_scale="max"
            ):
                mod = relay.quantize.quantize(mod, params, dataset=input_data)
        else:
            with relay.quantize.qconfig(
                calibrate_mode="global_scale", global_scale=8.0
            ):
                mod = relay.quantize.quantize(mod, params)
        return mod
