# ⛩️ LargeOracle Distillation App (WIP)
Adapt large models to your task and  extract a small and efficient model out of it.

If you like this App, give us a star to show your support for the project ⭐

## 📚 Description
The LargeOracle Distillation App takes a large model and a large semi-labeled dataset to automatically extract the task-relevant information and distill it into a smaller, more efficient model. The App uses advanced knowledge distillation techniques to train the smaller model on the task-specific labels extracted from the large model trained on a more generic task, ensuring that it retains the important information while being much easier to use and deploy.

During the labeled-data-generation phase, the large model is used to extract the task labels from the semi-labeled dataset. Once this is done, the large model is no longer needed and the smaller, distilled model can be used for inference.

Overall, LargeOracle Distillation allows users to leverage the power of large models without the associated complexity and resource requirements. Try it out today, and reach out if you have any feedback!