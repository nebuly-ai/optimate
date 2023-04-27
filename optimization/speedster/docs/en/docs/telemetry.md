# Telemetry


`Speedster` is a young and rapidly evolving open-source project. There is plenty of room for improvement for Speedster to make your model achieve the very best performance on your hardware... and you may still find some bugs in the code 🪲

Contributions to this OSS project are warmly welcomed 🤗. We encourage you to check out the Contribution guidelines to understand how you can become an active contributor of the source code.

## Sharing feedback to improve Speedster

Open source is a unique resource for sharing knowledge and building great projects collaboratively with the OSS community. To support the continued development, upon installation of Speedster you could share the information strictly necessary to improve the performance of this open-source project and facilitate bug detection and fixing.

More specifically, you will foster project enhancement by sharing details of the optimization techniques used with Speedster and the performance achieved on your model and hardware.

**Which data do we collect?**

We make sure to collect as little data as possible to improve the open-source project:

- basic information about the environment
- basic information about the optimization

Please find below an example of telemetry collection:

```python
{
"nebullvm_version": "0.6.0",
"app_version": "0.0.1",
"model_id": "e33a1bbf-fcfd-4f5a-81c9-a9154c7e9343_-7088971112344091114",
"model_metadata": {
    "model_name": "ResNet",
    "model_size": "102.23 MB",
    "framework": "torch"
},
"hardware_setup": {
    "cpu": "Apple M1 Pro",
    "operative_system": "Darwin",
    "ram": "17.18 GB"
},
"optimizations": [
    {
        "compiler": "torch",
        "technique": "original",
        "latency": 0.03
    },
    {
        "compiler": "NUMPY_onnxruntime",
        "technique": "none",
        "latency": 0.01
    }
],
"ip_address": "1.1.1.1"
}
```

**How to opt-out?**

You can simply opt-out from telemetry collection by setting the environment variable `SPEEDSTER_DISABLE_TELEMETRY to 1`.

**Should I opt out?**

Being open-source, we have very limited visibility into the use of the tool unless someone actively contacts us or opens an issue on GitHub.

We would appreciate it if you would maintain telemetry, as it helps us improve the source code. In fact, it brings increasing value to the project and helps us to better prioritize feature development.

We understand that you may still prefer not to share telemetry data and we respect that desire. Please follow the steps above to disable data collection.