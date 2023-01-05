# 🐉 OpenAlphaTensor App (WIP)
OpenAlphaTensor provides an open-source implementation of Deepmind's AlphaTensor algorithm.

With OpenAlphaTensor, you can increase the computational performances of an AI model with custom-generated matrix multiplication algorithms. You can train your own AlphaTensor algorithm for a specific matrix size or fine-tune a pre-trained AlphaTensor model to produce optimized kernels for a specific hardware.


OpenAlphaTensor is based on Deepmind's paper [Discovering Faster Matrix Multiplication Algorithms with Reinforcement Learning](https://www.nature.com/articles/s41586-022-05172-4).

If you appreciate the project, show it by [leaving a star ⭐](https://github.com/nebuly-ai/nebullvm/stargazers)


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
