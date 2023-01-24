# **Jupyter notebooks**

This folder contains notebooks showing how to use the `Speedster` app to optimize several models. 

The following frameworks are supported:
- PyTorch
- HuggingFace
- Tensorflow
- ONNX

Examples of how to use `Speedster` are shown for each of these frameworks.

In each folder we provide links to google colab where you can easily test the notebooks. 
If you want to test them on your own hardware, you can follow the guide below.

## 1. Setup
To test notebooks, we have to create an environment where all the required dependencies are installed.

First of all, clone the `nebullvm` repository:
```
git clone https://github.com/nebuly-ai/nebullvm.git
```
Next, navigate to the repo's root directory:
```
cd nebullvm
```

After cloning the repository there are two options: we can either install `Speedster` in a local environment or use a ready-to-use docker container.

### a. Using a local environment

Install `Speedster` library:
```
pip install speedster
```

Install deep learning compilers:
```
python -m nebullvm.installers.auto_installer \
    --frameworks all --compilers all
```

You can find additional options and details on the official [installation guide](https://docs.nebuly.com/modules/speedster/installation).

After everything has been installed, you can start a jupyter session with the following command:

```
jupyter notebook --allow-root --port 8888
```
And navigate a web browser to the IP address or hostname of the host machine at port 8888: `http://[host machine]:8888`

Use the token listed in the output from running the jupyter command to log in, for example:

`http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b`

You can finally navigate to the `notebooks/speedster` folder and then to the folder of the framework that you want to try and start a notebook.


### b. Using a Docker container

Another very easy way to test the following notebooks is by using one of the docker containers released on [dockerhub](https://hub.docker.com/r/nebulydocker/nebullvm). 


Pull the most up-to-date container image that has all compilers and their dependencies preinstalled:
```
docker pull nebulydocker/nebullvm:latest
```
Once pulled, the container can be launched with the following command:
```
docker run --rm --gpus all -ti -p 8888:8888 -v $PWD:/nebullvm nebulydocker/nebullvm:latest
```
The `-v` option in the command above allows to persist all the changes that will be done to the notebooks inside the container.
Please note that, in order to enable gpu inside docker, you have to ensure that nvidia docker is installed. Please follow the "Setting up NVIDIA Container Toolkit" part from the 
official [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
You can then check that the gpu can be seen inside the container by running `nvidia-smi` inside it, and checking that your gpu appears in the output.

Inside the container, we can then navigate to the notebooks folder:
```
cd /nebullvm/notebooks/speedster
```
We can then run a jupyter session with the following command:
```
jupyter notebook --allow-root --ip 0.0.0.0 --port 8888
```
And navigate a web browser to the IP address or hostname of the host machine at port 8888: `http://[host machine]:8888`

Use the token listed in the output from running the jupyter command to log in, for example:

`http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b`

You can finally navigate to the folder of the framework that you want to try and start a notebook.

## 2. Contributions
At Nebuly we are always eager to see how our library manages to optimise more and more models. If you test nebullvm on your model and this is not already present among the notebooks, feel free to open a PR for us to add your notebook to the repository!
