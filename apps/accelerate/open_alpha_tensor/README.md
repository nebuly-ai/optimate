# 🐉 OpenAlphaTensor App (WIP)
Boost your DL model's performance with OpenAlphaTensor's custom-generated matrix multiplication algorithms.

If you like this App, give us a star to show your support for the project ⭐

## 📖 Description
The OpenAlphaTensor App provides an open-source implementation of Deepmind's AlphaTensor algorithm for matrix multiplication. The algorithm is based on the paper [Discovering faster matrix multiplication algorithms with reinforcement learning](https://www.nature.com/articles/s41586-022-05172-4).

The App is easy to use: you can either train your own model for a specific matrix size or fine-tune an existing one for producing kernels optimized for a target hardware.

## 🚀 Get started
For training your AlphaTensor model, you can exectute the following command:
```bash
python main.py 
```

Model parameters can be given either as command line arguments or as a JSON file. The `config.json` file contains the default parameters for training a model for matrix size 4x4x4.

## 🧪 Missing features
- [ ] Add support for fine-tuning on target hardware.
- [ ] Release Pre-trained models.
- [ ] Support training on Multiple GPUs (it allows training on a larger batch size).
- [ ] Reduce memory footprint of the Acting Agent.
- [ ] Improve acting speed.
