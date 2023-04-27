# 🐉 OpenAlphaTensor
OpenAlphaTensor provides an open-source implementation of Deepmind's AlphaTensor algorithm.

With OpenAlphaTensor, you can increase the computational performances of an AI model with custom-generated matrix multiplication algorithms. You can train your own AlphaTensor algorithm for a specific matrix size or fine-tune a pre-trained AlphaTensor model to produce optimized kernels for a specific hardware.

OpenAlphaTensor is based on Deepmind's paper [Discovering Faster Matrix Multiplication Algorithms with Reinforcement Learning](https://www.nature.com/articles/s41586-022-05172-4).

If you appreciate the project, show it by [leaving a star ⭐](https://github.com/nebuly-ai/nebullvm/stargazers)

## 🧑‍🏫 Installation
You can install the package cloning the repository and running the following commands:
```bash
git clone https://github.com/nebuly-ai/nebullvm.git
cd nebullvm/apps/accelerate/open_alpha_tensor
pip install -e .
```

## 🚀 Get started
For training your AlphaTensor model, you can execute the following command:
```bash
python main.py 
```
Model parameters can be given either as command line arguments or as a JSON file. The `config.json` file contains the default parameters for training a model for matrix size 4x4x4.

Alternatively, if you want to have a more fine-grained control over the training process, you can use the python API:
```python
from open_alpha_tensor import train_alpha_tensor

cardinality_vector = 5  # The actions can have values in range [-2, 2]
N_bar = 100  # parameter for smoothing the temperature while adjusting the probability distribution
matrix_size = 5
input_size = matrix_size**2
n_steps = 15
n_actions = cardinality_vector ** (3 * input_size // n_steps)
action_memory = 7

train_alpha_tensor(
    tensor_length=action_memory + 1,
    input_size=input_size,
    scalars_size=1,
    emb_dim=2048,
    n_steps=n_steps,
    n_logits=n_actions,
    n_samples=32,
    device="cuda",
    len_data=2048,
    n_synth_data=1000000,
    pct_synth=0.7,
    batch_size=32,
    epochs=600000,
    lr=1e-4,
    lr_decay_factor=0.5,
    lr_decay_steps=5000,
    weight_decay=1e-5,
    optimizer_name="adamw",
    loss_params=(1, 1),
    limit_rank=150,
    checkpoint_dir="path/to/checkpoint/dir",
    checkpoint_data_dir="path/where/to/save/data/generated/by/the/model",
    n_actors=1,
    mc_n_sim=200,
    n_cob=100000,
    cob_prob=0.9983,
    data_augmentation=True,
    N_bar=N_bar,
    random_seed=42,
    extra_devices=None,
    save_dir="path/to/save/final/model",
)
```

## 🧪 Missing features
- [ ] Release weights of pre-trained models. **Coming out soon**.
- [ ] Add compilation of Alpha Tensor kernels in OpenAI's Triton and JAX/XLA.
- [ ] Add support for fine-tuning on target hardware.
- [ ] Support training on Multiple GPUs (it allows training on a larger batch size).
- [ ] Add support for other compilers (e.g. llvm).
- [ ] Reduce memory footprint of the Acting Agent.
- [ ] Improve acting speed.

## 💫 Contributing

We welcome contributions of all kinds, including new features, improved infrastructure, and better documentation. If you're interested in contributing, please see the [linked](https://docs.nebuly.com/contributions) page for more information on how to get involved.

A special thanks to [BrianPulfer](https://github.com/BrianPulfer) for his awesome contribution to the OpenAlphaTensor module.
