from collections import Callable
from typing import Optional

import torch.nn

from nebullvm.base import DeepLearningFramework, ModelParams, QuantizationType
from nebullvm.inference_learners.pytorch import PytorchBackendInferenceLearner
from nebullvm.optimizers import BaseOptimizer
from nebullvm.optimizers.quantization.pytorch import quantize_torch
from nebullvm.optimizers.quantization.utils import (
    check_quantization,
    check_precision,
)
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager
from nebullvm.utils.onnx import convert_to_target_framework
from nebullvm.utils.torch import create_model_inputs_torch, run_torch_model


class PytorchBackendOptimizer(BaseOptimizer):
    """Optimizer working directly on the pytorch backend, with no need of a
    conversion to ONNX. The model will be finally compiled using torchscript.
    For avoiding un-wanted modification to the input model models are copied
    before being optimized.

    Attributes:
        logger (Logger, optional): Optional logger for logging optimization
            information.
    """

    def optimize(
        self,
        model: torch.nn.Module,
        output_library: DeepLearningFramework,
        model_params: ModelParams,
        input_tfms: MultiStageTransformation = None,
        metric_drop_ths: float = None,
        quantization_type: QuantizationType = None,
        metric: Callable = None,
        input_data: DataManager = None,
    ) -> Optional[PytorchBackendInferenceLearner]:
        """Optimize the input model using pytorch built-in techniques.

        Args:
            model (torch.nn.Module): The pytorch model. For avoiding un-wanted
                modifications to the original model, it will be copied in the
                method.
            output_library (DeepLearningFramework): Output framework. At the
                current stage just PYTORCH is supported.
            model_params (ModelParams): Model parameters.
            input_tfms (MultiStageTransformation, optional): Transformations
                to be performed to the model's input tensors in order to
                get the prediction.
            metric_drop_ths (float, optional): Threshold for the accepted drop
                in terms of precision. Any optimized model with an higher drop
                will be ignored.
            quantization_type (QuantizationType, optional): The desired
                quantization algorithm to be used.
            metric (Callable, optional): If given it should
                compute the difference between the quantized and the normal
                prediction.
            input_data (DataManager, optional): User defined data.

        Returns:
            PytorchBackendInferenceLearner: Model optimized for inference.
        """
        self._log(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )
        assert output_library is DeepLearningFramework.PYTORCH, (
            "Other APIs than the Pytorch one are not supported "
            "for the Pytorch Backend yet."
        )
        check_quantization(quantization_type, metric_drop_ths)
        if metric_drop_ths is not None:
            if input_data is None:
                input_data_torch = [
                    tuple(
                        create_model_inputs_torch(
                            model_params.batch_size, model_params.input_infos
                        )
                    )
                ]
            else:
                input_data_torch, ys = input_data.get_numpy_list(
                    300, with_ys=True
                )
                input_data_torch = [
                    tuple(
                        convert_to_target_framework(t, output_library)
                        for t in data_tuple
                    )
                    for data_tuple in input_data_torch
                ]
            output_data_torch = [
                tuple(run_torch_model(model, list(input_tensors)))
                for input_tensors in input_data_torch
            ]
            model, input_tfms = quantize_torch(
                model, quantization_type, input_tfms, input_data_torch
            )

        learner = PytorchBackendInferenceLearner.from_torch_model(
            model,
            network_parameters=model_params,
            input_tfms=input_tfms,
            input_data=list(input_data.get_list(1)[0])
            if input_data is not None
            else None,
        )
        if metric_drop_ths is not None:
            is_valid = check_precision(
                learner,
                input_data_torch,
                output_data_torch,
                metric_drop_ths,
                metric_func=metric,
                ys=ys,
            )
            if not is_valid:
                return None
        return learner
