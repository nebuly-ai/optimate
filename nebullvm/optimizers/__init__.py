from typing import Dict, Type

from nebullvm.base import ModelCompiler
from nebullvm.optimizers.base import BaseOptimizer
from nebullvm.optimizers.openvino import OpenVinoOptimizer
from nebullvm.optimizers.tensor_rt import TensorRTOptimizer
from nebullvm.optimizers.tvm import ApacheTVMOptimizer

COMPILER_TO_OPTIMIZER_MAP: Dict[ModelCompiler, Type[BaseOptimizer]] = {
    ModelCompiler.TENSOR_RT: TensorRTOptimizer,
    ModelCompiler.APACHE_TVM: ApacheTVMOptimizer,
    ModelCompiler.OPENVINO: OpenVinoOptimizer,
}
