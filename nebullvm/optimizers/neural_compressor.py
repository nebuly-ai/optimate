import logging
from copy import deepcopy
from typing import Optional, Callable, Any

import torch

from nebullvm.base import ModelParams, DeepLearningFramework, QuantizationType
from nebullvm.config import CONSTRAINED_METRIC_DROP_THS
from nebullvm.inference_learners.neural_compressor import (
    NEURAL_COMPRESSOR_INFERENCE_LEARNERS,
    NeuralCompressorInferenceLearner,
)
from nebullvm.measure import compute_relative_difference
from nebullvm.optimizers import BaseOptimizer
from nebullvm.optimizers.quantization.neural_compressor import (
    quantize_neural_compressor,
)
from nebullvm.optimizers.quantization.utils import (
    check_quantization,
    check_precision,
)
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.utils.data import DataManager


class NeuralCompressorOptimizer(BaseOptimizer):
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
    ) -> Optional[NeuralCompressorInferenceLearner]:
        """Optimize the input model using Intel Neural Compressor Quantization.

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
            NeuralCompressorInferenceLearner: Model optimized for inference.
        """
        self._log(
            f"Optimizing with {self.__class__.__name__} and "
            f"q_type: {quantization_type}."
        )

        assert output_library is DeepLearningFramework.PYTORCH, (
            "Other APIs than the Pytorch one are not supported "
            "for the Neural Compressor yet."
        )
        check_quantization(quantization_type, metric_drop_ths)

        if quantization_type is None:
            return None

        train_input_data = input_data.get_split("train")

        model_quant = quantize_neural_compressor(
            deepcopy(model), quantization_type, train_input_data
        )

        learner = NEURAL_COMPRESSOR_INFERENCE_LEARNERS[output_library](
            input_tfms=input_tfms,
            network_parameters=model_params,
            model=model,
            model_quant=model_quant,
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
                    "The model optimized with neural compressor gives a "
                    "different result compared with the original model. "
                    "This compiler will be skipped.",
                    level=logging.WARNING,
                )
            return None
        return learner
