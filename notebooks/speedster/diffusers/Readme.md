# **Diffusers Optimization**

This section contains all the available notebooks that show how to leverage Speedster to optimize Diffusers models.

## Notebooks:
| Notebook                                                                                                                                                                             | Description                                                                        |                                                                                                                                                                                                                                             |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Accelerate Diffusers Stable Diffusion](https://github.com/nebuly-ai/nebullvm/blob/main/notebooks/speedster/diffusers/Accelerate_StableDiffusion_with_Speedster.ipynb) | Show how to optimize with Speedster the Stable Diffusion models from Diffusers. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nebuly-ai/nebullvm/blob/main/notebooks/speedster/diffusers/Accelerate_StableDiffusion_with_Speedster.ipynb) |

## Diffusers API quick view:

``` python
import torch
from speedster import optimize_model
from diffusers import StableDiffusionPipeline


# Load Stable Diffusion 1.4 as example
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    # On GPU we load by default the model in half precision, because it's faster and lighter.
    pipe = StableDiffusionPipeline.from_pretrained(model_id, revision='fp16', torch_dtype=torch.float16)
else:
    pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Create some example input data
input_data = [
    "a photo of an astronaut riding a horse on mars",
    "a monkey eating a banana in a forest",
    "white car on a road surrounded by palm trees",
    "a fridge full of bottles of beer",
    "madara uchiha throwing asteroids against people"
]

# Run Speedster optimization
optimized_model = optimize_model(
    model=pipe,
    input_data=input_data,
    optimization_time="unconstrained",
    ignore_compilers=["torch_tensor_rt", "tvm"],
    metric_drop_ths=0.1,
)

# Try the optimized model
res = optimized_model(test_prompt).images[0]

test_prompt = "futuristic llama with a cyberpunk city on the background"
```

## Setup TensorRT Plugins (Optional)
Official Source: https://github.com/NVIDIA/TensorRT/tree/main/demo/Diffusion

If you want to optimise Stable Diffusion on Nvidia GPUs, you need to install and use the TensorRT Plugins for getting the maximum speed-up. In our experiments, they improved the speed of the model compared to the basic version of TensorRT by more than 60%. 

The easiest option to use the plugins is to use the nebullvm docker image, where everything is preinstalled by default:

```
docker pull nebulydocker/nebullvm:latest
```

If you prefer instead to setup your environment manually, you can follow the following guide:

### Step 1: Download and extract TensorRT tar
Reference: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar
- [Download](https://developer.nvidia.com/tensorrt) the TensorRT tar file that matches the CPU architecture and CUDA version you are using.
- Extract in a folder <TRT_FOLDER> the tar file:
```
tar -xzvf TensorRT-8.5.3.1.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz
```
- Add the absolute path to the TensorRT lib directory to the environment variable LD_LIBRARY_PATH:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TRT_FOLDER_ABSOLUTE_PATH>/TensorRT-8.5.3.1/lib>
```

:warning: It could be necessary to add this path also to the PATH env variable, if you have issues in the following steps run also `export PATH=$PATH:<TRT_FOLDER_ABSOLUTE_PATH>/TensorRT-8.5.3.1/lib>`


### Step 2: Clone the TensorRT repository
```
git clone https://github.com/NVIDIA/TensorRT.git
cd TensorRT
git submodule update --init --recursive
```

### Step 3: Build TensorRT plugins library

```
mkdir -p build && cd build
cmake .. -DTRT_OUT_DIR=$PWD/out
cd plugin
make -j$(nproc)

export PLUGIN_LIBS="$PWD/build/out/libnvinfer_plugin.so"
export LD_PRELOAD=$PLUGIN_LIBS
```

You are now ready to use plugins to optimise stable diffusion!
