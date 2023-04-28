<p align="center">
<br><br><br>
<a https://docs.nebuly.com/welcome/quick-start"><img src="https://user-images.githubusercontent.com/42771598/235086376-5d859429-fd33-4019-a2a5-5a835e19d7cb.svg" width="400px"></a>
<br><br><br>
</p>

<p align="center">
<b>The next-generation platform to monitor and optimize your AI costs in one place</b>
</p>

<p align=center>
<a href="https://pypi.org/project/nebullvm/"><img src="https://badge.fury.io/py/nebullvm.svg"></a>
<a href="https://pypistats.org/packages/nebullvm"><img src="https://pepy.tech/badge/nebullvm"></a>
<a href="https://discord.gg/77d5kGSa8e"><img src="https://img.shields.io/badge/Discord-1.1k-blueviolet?logo=discord&amp;logoColor=white&style=round">
<a href="https://twitter.com/nebuly_ai"><img src="https://img.shields.io/twitter/url.svg?label=Follow%20%40nebuly_ai&style=social&url=https%3A%2F%2Ftwitter.com-nebuly_ai"></a>


</a>

`Nebuly` is the next-generation platform to monitor and optimize your AI costs in one place. The platform connects to all your AI cost sources (compute, API providers, AI software licenses, etc) and centralizes them in one place to give you full visibility and control. The platform also provides optimization recommendations and a co-pilot model that can guide during the optimization process. The platform builds on top of the open-source tools allowing you to optimize the different steps of your AI stack to squeeze out the best possible cost performances.

If you like the idea, give us a star to show your support for the project ⭐

*Apply for enterprise version early access here:* https://qpvirevo4tz.typeform.com/to/X7VfuRiH

## **AI costs monitoring (SDK)**

The monitoring platform allows you to monitor 100% of your AI costs. We support 3 main buckets of costs: 

- Infrastructure and compute (AWS, Azure, GCP, on-prem)
- AI-related software/tools licenses (OpenAI, Cohere, Scale AI, Snorkel, Pinecone, HuggingFace, Databricks, etc)
- People (Jira, GitLab, Asana, etc)

The easiest way to install the SDK is via `pip`:

```python
pip install nebuly
```
*The list of the supported integrations will be available soon*.

## **Cost optimization**

Once you have full visibility over your AI costs, you are ready to optimize them. We have developed multiple open-source tools to optimize the cost and improve the performances of your AI systems: 

✅ [Speedster](https://github.com/nebuly-ai/nebuly/tree/main/optimization/speedster): reduce inference costs by leveraging SOTA optimization techniques that best couple your AI models with the underlying hardware (GPUs and CPUs)

✅ [Nos](https://github.com/nebuly-ai/nos): reduce infrastructure costs by leveraging real-time dynamic partitioning and elastic quotas to maximize the utilization of your Kubernetes GPU cluster

✅ [ChatLLaMA](https://github.com/nebuly-ai/nebuly/tree/main/optimization/chatllama): reduce hardware and data costs by leveraging fine-tuning optimization techniques and RLHF alignment

## Contributing
As an open source project in a rapidly evolving field, we welcome contributions of all kinds, including new features, improved infrastructure, and better documentation. If you're interested in contributing, please see the [linked](https://docs.nebuly.com/contributions) page for more information on how to get involved.

---

<p align="center">
  <a href="https://discord.gg/RbeQMu886J">Join the community</a> |
  <a href="https://docs.nebuly.com/contributions/">Contribute to the library</a>
</p>
