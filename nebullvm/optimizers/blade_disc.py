import logging
import warnings

from collections.abc import Callable
from typing import Optional, Any

import torch.nn

from nebullvm.base import DeepLearningFramework, ModelParams, QuantizationType
from nebullvm.config import (
    NO_COMPILER_INSTALLATION,
    QUANTIZATION_DATA_NUM,
    CONSTRAINED_METRIC_DROP_THS,
)
from nebullvm.inference_learners.blade_disc import BladeDISCInferenceLearner
from nebullvm.measure import compute_relative_difference
from nebullvm.optimizers import BaseOptimizer
from nebullvm.optimizers.quantization.pytorch import quantize_torch
from nebullvm.optimizers.quantization.utils import (
    check_quantization,
    check_precision,
)
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager
from nebullvm.utils.torch import create_model_inputs_torch

try:
    import torch_blade
except ImportError:
    # TODO: Remove the False flag for allowing BladeDISC to be installed by
    # the Auto-Installer.
    if False and not NO_COMPILER_INSTALLATION:
        warnings.warn(
            "No valid BladeDISC installation has been found. "
            "Trying to re-install it from source."
        )
        from nebullvm.installers.installers import install_bladedisc

        install_bladedisc()
        import torch_blade
    else:
        warnings.warn(
            "No BladeDISC library detected. "
            "The BladeDISC Inference learner should not be used."
        )


class BladeDISCOptimizer(BaseOptimizer):
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
        model_outputs: Any = None,
    ) -> Optional[BladeDISCInferenceLearner]:
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
                get the prediction. Default: None.
            metric_drop_ths (float, optional): Threshold for the accepted drop
                in terms of precision. Any optimized model with an higher drop
                will be ignored. Default: None.
            quantization_type (QuantizationType, optional): The desired
                quantization algorithm to be used. Default: None.
            metric (Callable, optional): If given it should
                compute the difference between the quantized and the normal
                prediction. Default: None.
            input_data (DataManager, optional): User defined data.
                Default: None.
            model_outputs (Any, optional): Outputs computed by the original
                model. Default: None.

        Returns:
            BladeDISCInferenceLearner: Model optimized for inference.
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

        train_input_data = input_data.get_split("train").get_list(
            QUANTIZATION_DATA_NUM
        )

        if quantization_type is not None:
            model, input_tfms = quantize_torch(
                model, quantization_type, input_tfms, train_input_data
            )

        with torch.no_grad():
            model = torch_blade.optimize(
                model,
                allow_tracing=True,
                model_inputs=tuple((input_data.get_list(1)[0]))
                if input_data is not None
                else tuple(
                    create_model_inputs_torch(
                        model_params.batch_size, model_params.input_infos
                    )
                ),
            )

        learner = BladeDISCInferenceLearner.from_torch_model(
            model,
            network_parameters=model_params,
            input_tfms=input_tfms,
            input_data=list(input_data.get_list(1)[0])
            if input_data is not None
            else None,
        )

        test_input_data, ys = input_data.get_split("test").get_list(
            with_ys=True
        )

        is_valid = check_precision(
            learner,
            test_input_data,
            model_outputs,
            metric_drop_ths
            if quantization_type is not None
            else CONSTRAINED_METRIC_DROP_THS,
            metric_func=metric
            if quantization_type is not None
            else compute_relative_difference,
            ys=ys,
        )
        if not is_valid:
            if quantization_type is None:
                self._log(
                    "The model optimized with blade_disc gives a "
                    "different result compared with the original model. "
                    "This compiler will be skipped.",
                    level=logging.WARNING,
                )
            return None
        return learner
