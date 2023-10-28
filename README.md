# Nebuly - The User Analytics Platform for LLMs

This is the official Python SDK for Nebuly. **[Nebuly](https://www.nebuly.com/)** is the user analytics platform for LLMs enabling you to automatically capture how your users interact with your models. The platform helps you understand what your LLM users like, what they don’t and why, what are the most asked questions and how you can improve your LLMs products to delight your customers.

## Installation

The easiest way to install Nebuly’s SDK is via `pip`:

```
pip install nebuly
```

Once installed, authenticate to Nebuly platform and start building.

## Get started

Tracking interactions to Nebuly is incredibly easy and requires just two lines of code. Use your LLMs as always, all you need to do is:

1. Include Nebuly's API key.
2. Add the **`user_id`** parameter within your model call.

With these simple additions, you can start tracking interactions to Nebuly in less than 2 minutes.

```python
import nebuly
import openai

nebuly.init(api_key="<nebuly_api_key>")
openai.api_key = "<your_openai_api_key>"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You are an helpful assistant"
        },
        {
            "role": "user",
            "content": "Hello, I need help with my computer"
        }
    ],
    user_id="test_user",
)
```

We support also Azure OpenAI, HuggingFace, Cohere, Anthropic, VertexAI and Bedrock. To learn more please visit the [documentation](https://docs.nebuly.com/welcome/overview).
