# 🛰️ GPU Partitioner (WIP)
Effortlessly maximize the utilization of GPU resources in a Kubernetes cluster through real-time dynamic partitioning and elastic quotas.

If you like this modeule, give us a star to show your support for the project ⭐

## 📚 Description
The GPU Partitioner module is a Kubernetes plug-in that allows users to easily schedule Pods requesting fractions of GPUs without having to manually partition them. The plug-in uses a dynamic partitioning system that monitors the GPU resources of the cluster in real time. The App  determines the optimal partitioning of the available GPUs to maximize their utilization.

The GPU Partitioner component is a bit like a Cluster Autoscaler for GPUs, allowing users to dynamically partition their GPUs instead of scaling up the number of nodes and GPUs. The actual partitioning of the GPUs is carried out by agents deployed on every eligible node of the cluster, which expose their partitioning state to the GPU Partitioner and apply the desired partitioning state decided by the plug-in.

Overall, the GPU Partitioner AI module provides a simple and effective way to maximize the utilization of GPU resources in a Kubernetes cluster, allowing users to schedule more Pods and get the most out of their GPUs. Try it out today, and reach out if you have any feedback!
