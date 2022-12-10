# 💥 AutoBoost app (WIP)

Automatically apply SOTA optimization techniques to achieve the maximum inference speed-up on your hardware.


## 📖 Description
The AutoBoost App leverages state-of-the-art open-source optimization tools to enable the acceleration of the entire software-to-hardware stack. AutoBoost requires little to no knowledge of optimization techniques and, as such, is the ideal starting point for any ML engineer looking for extra performance. The workflow consists of 3 steps: select, search, and serve.

In the select step, users input their model in their preferred deep learning framework and express their preferences regarding maximum consented accuracy loss and optimization time. This information is used to guide the optimization process and ensure that the resulting model meets the user's needs.

In the search step, the library automatically tests every combination of optimization techniques across the software-to-hardware stack, such as sparsity, quantization, and compilers, that is compatible with the user's preferences and hardware. This allows the library to find the optimal configuration of techniques for accelerating the model.

Finally, in the serve step, the library returns an accelerated version of the user's model in the DL framework of choice, providing a significant boost in performance. 

Overall, AutoBoost provides a powerful and easy-to-use tool for optimizing AI models and getting the most out of the software-to-hardware stack. Try it out today, and reach out if you have any feedback!