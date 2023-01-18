import copy
import os

import numpy as np
import pytest
import requests
import torch
import types
from PIL import Image
from tempfile import TemporaryDirectory

from speedster.utils import save_yolov5_model, load_yolov5_model, OptimizedYolo
from speedster import optimize_model


def test_yolov5_save_and_load():
    # Images
    imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images
    img_name = "zidane.png"
    Image.open(requests.get(imgs[0], stream=True).raw).save(img_name)
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)
    core_model = copy.deepcopy(yolo_model.model.model)
    core_model._forward_once = types.MethodType(_forward_once, core_model)
    list_of_layers = list(core_model.model.children())
    last_layer = list_of_layers.pop(-1)
    core_model.model = torch.nn.Sequential(*list_of_layers)
    core_wrapper = CoreModelWrapper(core_model, last_layer.f)
    input_data = [((read_and_crop(img_name, core_model, (384, 640)),), None) for _ in range(500)]
    model_optimized = optimize_model(
        model=core_wrapper,
        input_data=input_data,
        optimization_time="unconstrained",
        metric_drop_ths=3
    )
    final_core = OptimizedYolo(model_optimized, last_layer)
    yolo_model.model.model = final_core
    with TemporaryDirectory() as tmp_dir:
        save_yolov5_model(yolo_model, tmp_dir)
        loaded_model = load_yolov5_model(tmp_dir)

        size_saved = yolo_model.model.model.get_size()
        size_loaded = loaded_model.model.model.get_size()

        assert isinstance(loaded_model.model.model, OptimizedYolo)
        assert ((size_loaded - size_loaded * 0.002) < size_saved < (size_loaded + size_loaded * 0.002))


def _forward_once(self, x, profile=False, visualize=False):
    y, dt = [], []  # outputs
    for m in self.model:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        if profile:
            self._profile_one_layer(m, x, dt)
        x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output
        if visualize:
            feature_visualization(x, m.type, m.i, save_dir=visualize)
    self.last_y = y
    return x

def read_and_crop(im, original_model, img_size):
    p  =  next(original_model.parameters())
    im = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im)
    max_y, max_x = im.size
    ptr_x = np.random.choice(max_x-img_size[0])
    ptr_y = np.random.choice(max_y-img_size[1])
    im = np.array(im.crop((ptr_y, ptr_x, ptr_y + img_size[1], ptr_x + img_size[0])))
    x = np.expand_dims(im, axis=0)
    x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
    x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
    return x

class CoreModelWrapper(torch.nn.Module):
    def __init__(self, core_model, output_idxs):
        super().__init__()
        self.core = core_model
        self.idxs = output_idxs

    def forward(self, *args, **kwargs):
        x = self.core(*args, **kwargs)
        return tuple(x if j == -1 else self.core.last_y[j] for j in self.idxs)
