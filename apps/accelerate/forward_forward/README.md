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

Currently three architectures are supported:
* `progressive`: the most simple architecture described in the paper. It has a pipeline-like structure and each layer can be trained independently from the following ones. Our implementation differs respect the original one since the labels are injected in the image concatenating them to the flattened tensor instead of replacing the first n_classes pixels value with a one-hot-representation of the label.

* `recurrent`: the recurrent architecture described in the paper. It has a recurrent-like structure and its based on the `GLOM` architecture proposed by Hinton. 

* `nlp`: A simple network which can be used as a language model.

The recurrent and nlp network architectures are better explained below.

## Recurrent Architecture
The recurrent architecture is based in the `GLOM` architecture for videos, proposed by Hinton in the paper [How to represent part-whole hierarchies in a neural network](https://arxiv.org/pdf/2102.12627.pdf). Its application to the forward-forward algorithm aims at enabling each layer to learn not just from the previous layer output, but from the following layers as well. This is done by concatenating the outputs of the previous layer and following layers computed at the previous time-step. A learned representation of the label (positive or negative) it is given as input to the last layer. The following figure shows the structure of the network:

<img width="1216" alt="recurrent_net" src="https://user-images.githubusercontent.com/38586138/208651417-498c4bd4-f2dc-4613-a376-0b69317c73d4.png">

## NLP Architecture
The forward-forward architecture developed for NLP is a simple network which can be used as a language model. The network is composed by few normalized fully connected layers followed by a ReLU activation. All hidden representations are then concatenated together and given as input to the softmax for predicting the next token. The network can be trained in a progressive way, i.e. each layer can be sequentially trained separately from the following ones. The following figure shows the structure of the network:
<img width="666" alt="nlp_net" src="https://user-images.githubusercontent.com/38586138/208651624-c159b230-f903-4e13-aaa7-b39a0d1c52fc.png">

## What is missing
This app implements the main architectures exposed by hinton in its paper. However, there are still some features that are not implemented yet. In particular, the following features are missing:

* [ ] Implement unsupervised training.
* [ ] Implementation of the `progressive` architecture using local receptive fields instead of fully connected layers.
* [ ] Add training on CIFAR-10 for CV-based architectures.
