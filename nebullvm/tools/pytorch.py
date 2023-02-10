from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Sequence

from loguru import logger

from nebullvm.optional_modules.torch import torch, Module, DataLoader
from nebullvm.tools.base import DataType, InputInfo, Device, DeviceType
from nebullvm.tools.data import DataManager

FX_MODULE_NAME = "NebullvmFxModule"


def save_with_torch_fx(model: Module, path: Path):
    traced_model = torch.fx.symbolic_trace(model)
    traced_model.to_folder(path, FX_MODULE_NAME)


def load_with_torch_fx(
    path: Path, state_dict_name: str = "pruned_state_dict.pt"
):
    module_file = path / "module.py"
    with open(module_file, "r") as f:
        module_str = f.read()
    exec(module_str, globals())
    model = eval(FX_MODULE_NAME)()
    model.load_state_dict(torch.load(path / state_dict_name))
    return model


def get_outputs_sizes_torch(
    torch_model: Module,
    input_tensors: List[torch.Tensor],
    device: Device,
) -> List[Tuple[int, ...]]:
    if device.type is DeviceType.GPU:
        input_tensors = [x.to(device.to_torch_format()) for x in input_tensors]
        torch_model.to(device.to_torch_format())
    with torch.no_grad():
        outputs = torch_model(*input_tensors)
        if isinstance(outputs, torch.Tensor):
            return [tuple(outputs.size())]
        else:
            return [tuple(output.size()) for output in outputs]


def create_model_inputs_torch(
    input_infos: List[InputInfo],
) -> List[torch.Tensor]:
    input_tensors = (
        torch.randn(*input_info.size)
        if input_info.dtype is DataType.FLOAT32
        else torch.randint(
            size=input_info.size,
            low=input_info.min_value or 0,
            high=input_info.max_value or 100,
        )
        for input_info in input_infos
    )
    return list(input_tensors)


def run_torch_model(
    torch_model: Module,
    input_tensors: List[torch.Tensor],
    device: Device,
    dtype: torch.dtype = torch.float,
) -> List[torch.Tensor]:
    torch_model.eval()
    if device.type is DeviceType.GPU:
        torch_model.to(device.to_torch_format())
        if dtype != torch.half:
            input_tensors = (
                t.to(device.to_torch_format()) for t in input_tensors
            )
        else:
            input_tensors = (
                t.to(device.to_torch_format()).half()
                if t.dtype == torch.float
                else t.to(device.to_torch_format())
                for t in input_tensors
            )
    with torch.no_grad():
        pred = torch_model(*input_tensors)
    if isinstance(pred, torch.Tensor):
        pred = [pred.cpu()]
    else:
        pred = [p.cpu() for p in pred]
    return pred


def _extract_dynamic_axis(
    torch_model: Module,
    dataloader: DataManager,
    input_sizes: List[Tuple[int, ...]],
    device: Device,
    max_data: int = 100,
) -> Optional[Dict]:
    from nebullvm.tools.utils import inspect_dynamic_size

    dynamic_axis = {"inputs": [{}] * len(input_sizes), "outputs": []}
    output_sizes = []
    for i, input_data in enumerate(dataloader):
        input_tensors = input_data[0]
        if i >= max_data:
            break
        inspect_dynamic_size(
            input_tensors, input_sizes, dynamic_axis["inputs"]
        )
        outputs = tuple(run_torch_model(torch_model, input_tensors, device))
        if i == 0:
            dynamic_axis["outputs"] = [{}] * len(outputs)
            output_sizes = [tuple(output.shape) for output in outputs]
        inspect_dynamic_size(outputs, output_sizes, dynamic_axis["outputs"])
    if any(
        len(x) > 0 for x in (dynamic_axis["inputs"] + dynamic_axis["outputs"])
    ):
        return dynamic_axis
    return None


def extract_info_from_torch_data(
    model: Module,
    dataloader: Union[DataLoader, Sequence],
    dynamic_axis: Dict,
    device: Device,
):
    from nebullvm.tools.utils import ifnone

    input_data = (
        dataloader[0]
        if isinstance(dataloader, Sequence)
        else next(iter(dataloader))
    )
    input_row = input_data[0]
    batch_size = int(input_row[0].shape[0])
    if not all([input_row[0].shape[0] == x.shape[0] for x in input_row]):
        logger.warning("Detected not consistent batch size in the inputs.")

    input_sizes = [tuple(x.shape) for x in input_row]
    input_types = [
        "int64"
        if isinstance(x.cpu(), torch.LongTensor)
        else "int32"
        if isinstance(x.cpu(), torch.IntTensor)
        else "float32"
        for x in input_row
    ]

    if dynamic_axis is not None:
        dynamic_axis["inputs"] = [
            {int(k): v for (k, v) in val.items()}
            for val in dynamic_axis["inputs"]
        ]
        dynamic_axis["outputs"] = [
            {int(k): v for (k, v) in val.items()}
            for val in dynamic_axis["outputs"]
        ]

    dynamic_axis = ifnone(
        dynamic_axis,
        _extract_dynamic_axis(model, dataloader, input_sizes, device),
    )
    return batch_size, input_sizes, input_types, dynamic_axis


def torch_is_gpu_available():
    return torch.cuda.is_available()


def torch_get_device_name():
    return torch.cuda.get_device_name(0)
