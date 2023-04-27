# ⚡ LargeSpeedster App (WIP)
Automatically apply SOTA optimization techniques on large AI models to achieve the maximum acceleration on your hardware.

If you like this App, give us a star to show your support for the project ⭐

## 📚 Description
The LargeSpeedster App is a powerful tool to optimize large AI models (LMs). Leveraging state-of-the-art open-source optimization tools, LargeSpeedster enables the acceleration of large models, i.e. models with a number of parameters in excess of what could be stored on a single GPU. The workflow consists in 3 steps: select, search, and serve.

In the select step, users input their large model in their preferred deep learning framework and express their preferences regarding maximum consented accuracy loss. This information is used to guide the optimization process and ensure that the resulting model meets the user's needs.

In the search step, the App automatically tests multiple LMs-specific optimization techniques across the software-to-hardware stack, such as SmoothQuant quantization, FlashAttention, and inference-specific kernels. The App also tunes the optimal parallelization strategy and its configuration parameters, allowing it to find the optimal configuration of techniques for accelerating the model.

Finally, in the serve step, the App returns an accelerated version of the user's model in the DL framework of choice, providing a significant boost in performance.

Overall, LargeSpeedster is an easy-to-use tool that allows users to optimize their large AI models and get the most out of their software-to-hardware stack. Try it out today, and reach out if you have any feedback!
