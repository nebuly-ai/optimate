# Forward-Forward Algorithm

This app implements a complete open-source version of the forward forward algorithm recently presented by Geoffry Hinton.
The forward-forward algorithm is a method for training deep neural networks that is based on the idea of using two forward passes on positive and negative data for training the network. Differently respect to the backpropagation approach the forward forward do not need to compute the gradient of the loss function with respect to the parameters of the network. On the contrary each optimization step can be performed locally, meaning that each layer weights can be updated just after the layer has performed its own forward pass.

## Installation

The forward-forward app is built on top of nebullvm, a framework for efficiency based apps. The app can be easily installed from source code. First you have to clone the repository and navigate to the app directory:

```bash
git clone https://github.com/nebuly-ai/nebullvm.git
cd nebullvm/apps/accelerate/forward_forward
```

Then you can install the app with the following command:

```bash
pip install .
```
This process will just install the minimum requirements for running the app. If you want to run the app on a GPU you have to install the CUDA version of PyTorch. You can find the instructions on the official PyTorch website.

## Usage
At the current stage the app supports the main architectures discussed by Hinton in its paper. Each architecture can be trained with the following command:

```python
from forward_forward import train_with_forward_forward_algorithm


trained_model = train_with_forward_forward_algorithm(
    model_type="progressive",
    n_layers=3,
    hidden_size=2000,
    lr=0.03,
    device="cuda",
    epochs=100,
    batch_size=5000,
    theta=2.,
)
```

Currently the supported architectures are:
* `progressive`: the most simple architecture described in the paper. It has a pipeline-like structure and each layer can be trained independently from the following ones. Our implementation differs respect the original one since the labels are injected in the image concatenating them to the flattened tensor instead of replacing the first n_classes pixels value with a one-hot-representation of the label.

* `recurrent`: the recurrent architecture described in the paper. It has a recurrent-like structure and its based on the `GLOM` architecture proposed by Hinton. 

* `nlp`: A simple network which can be used as a language model.