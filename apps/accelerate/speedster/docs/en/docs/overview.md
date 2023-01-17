<<<<<<< HEAD
# Readme
=======
# Overview

`Speedster` is an open-source module designed to speed up AI inference in just a few lines of code. The library automatically applies the best set of SOTA optimization techniques to achieve the maximum inference speed-up (latency, throughput, model size) physically possible on your hardware (single machine).

`Speedster` makes it easy to combine optimization techniques across the whole software to hardware stack, delivering best in class speed-ups. If you like the idea, give us a star to support the project ⭐

![speedster_benchmarks](https://user-images.githubusercontent.com/42771598/212486740-431328f3-f1e5-47bf-b6c9-b6629399ad09.png)

The core `Speedster` workflow consists of 3 steps:


- [x]  **Select**: input your model in your preferred DL framework and express your preferences regarding:
    - Accuracy loss: do you want to trade off a little accuracy for much higher performance?
    - Optimization time: stellar accelerations can be time-consuming. Can you wait, or do you need an instant answer?
- [x]  **Search**: the library automatically tests every combination of optimization techniques across the software-to-hardware stack (sparsity, quantization, compilers, etc.) that is compatible with your needs and local hardware.
- [x]  **Serve**: finally, `Speedster` chooses the best configuration of optimization techniques and returns an accelerated version of your model in the DL framework of your choice (just on steroids 🚀).

Now you are ready to start accelerating your models, visit the (Installation)[] section to start right away!
>>>>>>> b903b9075f887f04b78a7a45cd7c1c018c4eb29e
