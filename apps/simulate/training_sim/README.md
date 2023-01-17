# 🏋️ TrainingSim (WIP)
Easily simulate and optimize the training of large AI models on a distributed infrastructure.

If you like this module, give us a star to show your support for the project ⭐

## 📖 Description
The TrainingSim module is a powerful tool for simulating the training behavior of large models on a distributed infrastructure. Unlike other solutions, it allows users to explore different parallelism strategies and configurations without the need for time-consuming and costly trial and error.

With TrainingSim, users can input their model and specify their accuracy requirements and computing budget. The module simulates multiple parallelism strategies, such as Tensor Parallelism and ZeRo parallelism, sweeping over different parameters and communication protocols. This allows it to find the optimal configuration for the given set of constraints, all without the need for actual training runs.

The module returns the configuration files that can be applied to the model using popular large model training frameworks such as DeepSpeed, MegatronLM, and ColossalAI. This allows users to quickly and easily set up their model for training, using the optimal configuration determined by the library without having to invest time and money on multiple training runs.

Overall, TrainingSim AI provides a fast and cost-effective way to simulate the training behavior of large models on a distributed infrastructure. It helps users to accelerate development times, optimize their AI systems and make the most of their budgets. Try it out today, and reach out if you have any feedback!
