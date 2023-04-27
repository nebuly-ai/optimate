# Overview


`Speedster` is an open-source module designed to accelerate AI inference in just a few lines of code.
The library allows you to seamlessy modulate the inference performances of your AI models in terms of latency, throughput, model size, accuracy, cost and automatically applies the best set of optimization techniques along the software to hardware stack to meet your targets.

`Speedster` makes it easy to combine optimization techniques across the whole software to hardware stack, delivering best in class speed-ups. If you like the idea, give us a star to support the project ⭐

![speedster](https://user-images.githubusercontent.com/53374883/225600620-1cd84073-d9b3-43d1-84fa-c3e6c25eb915.png)

The core `Speedster` workflow consists of 3 steps:


- [x]  **Select**: input your model in your preferred DL framework and express your preferences regarding:
    - Accuracy loss: do you want to trade off a little accuracy for much higher performance?
    - Optimization time: stellar accelerations can be time-consuming. Can you wait, or do you need an instant answer?
- [x]  **Search**: the library automatically tests every combination of optimization techniques across the software-to-hardware stack (sparsity, quantization, compilers, etc.) that is compatible with your needs and local hardware.
- [x]  **Serve**: finally, `Speedster` chooses the best configuration of optimization techniques and returns an accelerated version of your model in the DL framework of your choice (just on steroids 🚀).

Now you are ready to start accelerating your models, visit the [Installation](installation.md) section to start right away!
