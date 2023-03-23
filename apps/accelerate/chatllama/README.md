# **🦙 ChatLLaMA**

> :warning: Please note this library does NOT contain LLaMA’s weights; to access the weights, you need to apply to Meta's form.

`ChatLLaMA` 🦙 is a library that allows you to create hyper-personalized ChatGPT-like assistants using your own data and the least amount of compute possible. Instead of depending on one large assistant that “rules us all”, we envision a future where each of us can create our own personalized version of ChatGPT-like assistants. Imagine a future where many ChatLLaMAs at the "edge" will support a variety of human's needs. But creating a personalized assistant at the "edge" requires huge optimization efforts on many fronts: dataset creation, efficient training with RLHF, and inference optimization.

This library is meant to simplify the development of hyper-personalized ChatLLaMA assistants. Its purpose is to give developers peace of mind, by abstracting the efforts required for computational optimization and for the collection of large amounts of data.

If you like the project, please show your support by [leaving a star ⭐](https://github.com/nebuly-ai/nebullvm/stargazers).

## Quick install
You can install the package with pip:
```bash
pip install chatllama-py
```
Then you need to install the Llama models cloned from [Meta's repository](https://github.com/facebookresearch/llama):
```bash
git clone https://github.com/facebookresearch/llama.git
cd llama
pip install -r requirements.txt
pip install -e .
```
Follow the instructions in the Llama repository to download the model weights and tokenizer.

## What can ChatLLaMA help with?

`ChatLLaMA` 🦙 has been designed to help developers with various use cases, all related to RLHF training and optimized inference. These are some of the use cases that better resonate with our community wishlist:

- I want to create my personalized version of ChatGPT-like assistants for vertical specific tasks (legal, medical, gaming, academic research, etc.);
- I want to train an efficient ChatGPT-like assistant on my local hardware infrastructure using a limited amount of data;
- I want to create my own personalized version of ChatGPT-like assistant without costs getting out of control;
- I want to understand which model architecture (LLaMA, OPT, GPTJ, etc.) best fits my requirements in terms of hardware, compute budget, and performance;
- I want to align the assistant with my personal/company values, culture, brand and manifesto.

## Getting started

In this Getting Started we will set up a local RLHF training that will allow you to create your own ChatGPT-like assistant. In this example, we used OPT-1.3B, wherever possible we used open-source datasets and ran the training on a NVIDIA A100. If you want to use other models or hardware, we recommend reading the [supported models](#supported-models), [hardware requirements](#hardware-requirements) and [dataset preparation](#dataset-preparation) sections. In this example, we ran a few epochs of the training; this took a few hours. Any feedback on total training time, on any hardware, would be greatly appreciated. Please share your experience with our community on our Discord channel.

To quickly get you started, we will focus on 3 key steps:

1. Download YAML files to customize your training process. Please note that all the parameters of the library can be managed in the `config.yaml`;
2. Prepare the 3 datasets needed to train the actor model, the reward model and perform RLHF;
3. Train the models on your local infrastructure.

<details>
<summary>1 - YAML download </summary>
First, let’s get the artifacts for running ChatLLaMA. The artifacts contain:

- `config.yaml`: config file for model and data set. This allows you to 1) select the model you prefer (LLaMA, OPT, BLOOM, etc) 2) change all the hyperparameters of the training process;
- `ds_config.json`: config file to define DeepSpeed training parameters;
- `templates.json`: synthetic data generation templates that can be used to personalize the creation of the dataset. The templates are used for feeding LLMs during the data generation. Note that the `templates.json` file contains a dictionary having as *keys* the training steps (`actor`, `reward`, `rlhf`) and as *values* a string containing the personalization requests of the user. For more details see the [dataset preparation](#dataset-preparation) section;
- [`main.py`](http://main.py): file to train the model.
        
```bash
wget -O artifacts.zip https://nbllabartifacts.blob.core.windows.net/chatllama/artifacts.zip\?sp\=r\&st\=2023-03-08T14:53:24Z\&se\=2100-03-08T22:53:24Z\&spr\=https\&sv\=2021-06-08\&sr\=b\&sig\=jqr%2B2ZkR0SW9RjV0pDOdQ%2BDulLXLjbZ36vmNd4XxxyQ%3D
unzip artifacts.zip 
```
        
Once you have run the command above, you will find the all artificats in the `artifacts/` directory. Now you can move on to the next section regarding the dataset preparation.

</details>

<details>
<summary> 2 - Dataset preparation </summary>
    
Before training the model, we need to prepare 3 datasets:

- `actor_training_data`: this is the JSON dataset used in the supervised fine-tuning. It consists of examples of unlabelled conversations, e.g. collection of prompts and responses;
- `rlhf_training_data`: this is the JSON dataset used for RLHF training. It consists of a collection of possible input user prompts;
- `reward_training_data`: this is the JSON dataset used to train the reward model. It consists of responses with associated scores.

In this example, we are using only publicly available dataset and synthetic generation; if you want to use your own data instead, please see the [Dataset preparation](#dataset-preparation) section.

First, let’s download the `actor_training_data` and the `rlhf_training_data`: 

```bash
python artifacts/download_dataset.py ARLHF --path ./datasets --number_of_samples 200
```

Finally, let’s create the `reward_training_data` using `davinci-003` for synthetic data generation.

```bash
export OPENAI_API_KEY=YOUR_API_KEY
python artifacts/generate_rewards.py ./datasets/reward_training_data.json
```

> :warning: Creating the `reward_training_data` with `davinci-003` is not free, i.e. it costs a few $$. If you prefer avoiding external paid APIs, we suggest using HuggingFace’s models (e.g. flan_t5_xl) as described in more detail in the [Supported models](#supported-models) section.
> 
> :warning: if using OpenAI's API, please be aware of OpenAI's terms of use stating that it is forbidden to "use the Services to develop foundation models or other large scale models that compete with OpenAI".

At this point, we have successfully created the 3 datasets. We can therefore move on to the final section and start the training.

</details>

<details>
<summary> 3 - Training </summary>
    
You can train the 3 models in separate steps:

- Train the Reward Model

    ```bash
    python artifacts/main.py artifacts/config/config.yaml --type REWARD
    ```

- Pre-Train the Actor Model

    ```bash
    python artifacts/main.py artifacts/config/config.yaml --type ACTOR
    ```

- Training the Actor with reinforcement learning.

    ```bash
    python artifacts/main.py artifacts/config/config.yaml --type RL
    ```


or, equivantly, the 3 trainings can also be pipelined using the flag ALL.

```bash
python artifacts/main.py artifacts/config/config.yaml --type ALL
```

Note that the path to the datasets and the training hyper-parameters of the training process are specified in the `config.yaml` file.

</details>

## Contributing and Roadmap

As an open source project in a rapidly evolving field, we welcome contributions of all kinds, including new features, improved infrastructure, and better documentation. If you're interested in contributing, please see our [Roadmap page](https://github.com/users/nebuly-ai/projects/1/views/1) for more information on how to get involved.

You can participate in the following ways:

1. Submit an issue or PR on GitHub
2. Join our [Discord group](https://discord.gg/77d5kGSa8e) to chat

## Supported models

<details><summary><b><i> Actor models </i></b></summary>

We support models that can be run efficiently with a limited amount of compute, such as LLaMA and 🤗 transformers. These are the models with less than 20B parameters currently supported :

- LLaMA: 7B and 13B, please note this library does NOT contain LLaMA’s weights; to access the weights, you need to apply to Meta's [form](https://forms.gle/jk851eBVbX1m5TAv5).
- GPTJ: 6B
- GPTNeoX: 1.3B, 20B
- **(⚠️WIP)** Flan-T5: 80M, 259M, 780M, 3B, 11B
- OPT: 125M, 359M, 1.3B, 2.7B, 6.7B, 13B
- BLOOM: 560M, 1.1B, 1.7B, 3B, 7.1B
- BLOOMZ: 560M, 1.1B, 1.7B, 3B, 7.1B
- Galactica: 125M, 1.3B, 6.7B
</details>

<details><summary><b><i> Reward models </i></b></summary>

We suggest using models under 6B from 🤗 transformers: 

- GPT2: 124M, 355M, 774M, 1.5B
- OPT: 125M, 359M, 1.3B, 2.7B
- GPTJ: 6B
- BLOOMZ: 560M, 1.1B, 1.7B, 3B
- **(⚠️WIP)** OpenAssistant [pre-trained reward models](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2)
</details>

<details>
<summary><b><i> Synthetic data generation models </i></b></summary>

We support both APIs from OpenAI and  🤗 transformers:

- OpenAI: da-vinci-003, gpt-3.5-turbo **(⚠️WIP)**
- HuggingFace: Flan-T5 (3B and 11B)

> :warning: if using OpenAI's API, please be aware of OpenAI's terms of use stating that it is forbidden to "use the Services to develop foundation models or other large scale models that compete with OpenAI".

:watninh

If you need support for different models, please open an issue and we will get to work.
</details>

## Hardware requirements

<details><summary><b><i> Training </i></b></summary>

Larger actor models require more powerful hardware. Here is a rough hardware recommendation table, suggesting the right type of hardware for different actor model sizes:

- 125M to 1.3B → 1x Nvidia 3090/4090
- 1.3B to 3B → 1x Nvidia A100 (80Gb)
- 3B with DeepSpeed CPU off-loading → 1x Nvidia 3090/4090
- 3B to 7B with DeepSpeed ZeRO → 4x Nvidia T4
- 3B to 13B → 4x Nvidia A100 (80Gb)
- 13B to 20B with DeepSpeed ZeRO → 4x Nvidia A100 (80Gb)
- 13B to 20B → 8x Nvidia A100 (80Gb)
</details>

<details><summary><b><i> Inference </i></b></summary>

**(⚠️WIP)** When it comes to inference optimization, ChatLLaMA will support the following optimization techniques:

- [ ]  DeepSpeed ZeRO
- [ ]  FlexGen
- [ ]  HF Accelerate
- [ ]  PyTorch Vanilla
</details>

Please note that inference optimization has yet to be implemented. If you would like to contribute, please see the **issue roadmap**, community contributions are always welcome 😊.

## Dataset preparation

To successfully train a ChatLLaMA assistant, you need 3 different datasets: `actor_training_data`, `rlhf_training_data` and `reward_training_data`.

<details>
<summary> Dataset for supervised fine-tuning of the actor model </summary>
    
The `actor_training_data` is a collection of prompts with the associated responses as highlighted below:

```json
[
  {
      "user_input": "here the input of the user",
      "completion": "here the model completion"
  }
]
```

ChatLLaMA supports 4 different options to prepare the `actor_training_data`:

* <details><summary> Use 100% synthetic data </summary>

  The dataset can be synthetically generated by running the following command:

  ```bash
  python artifacts/generate_actor_dataset.py
  ```

  > :warning: Note that this command will require a subscription to OpenAI. Generating the full dataset with `davinci-003` could cost approximately ~200$.
  > 
  > :warning: if using OpenAI's API, please be aware of OpenAI's terms of use stating that it is forbidden to "use the Services to develop foundation models or other large scale models that compete with OpenAI".

  Alternatively, you can generate the dataset for free using 🤗 tranformers as described in the section [Supported models](#supported-models).
  </details>
  
* <details><summary> Use one of the open source datasets with assistant interactions </summary>

  Currently, we support:

  - [Anthropic HH RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf): this dataset consists of structured question/answer pairs with an LLM chatbot that includes selected and rejected answers;
  - [Stanford Human Preferences Dataset (SHP)](https://huggingface.co/datasets/stanfordnlp/SHP): this dataset is curated from selected "ask" subreddits, and includes questions that span a wide range of question/answer pairs based on the most upvoted responses. Please note that, unlike HH RLHF, this dataset is not intended to reduce harassment by selecting the ideal chatbot response, but instead weights the most helpful human responses.

  The datasets can be downloaded running the following command:

  ```bash
  python artifacts/download_dataset.py <dataset_name> --path <path_to_folder_for_download> --number_of_samples <N>
  ```

  Where: 

  - `<dataset_name>` could be "SHP" or "ARLHF" for the StanfordNLP/SHP dataset or ARLHF for the Anthropic/hh-rlhf dataset respectively;
  - `<path_to_folder_for_download>` is the folder path to where the datasets are going to be created;
  - `<N>` is the number of samples of which the reward_dataset.json is composed.
  </details>
  
  
* <details><summary> Use 100% personalized dataset </summary>

  The user provides his own personalized full dataset. Datasets must be JSON files with the following format:

  ```
  [
      {
          "user_input": "here the input of the user",
          "completion": "here the model completion"
      }
  ]
  ```

  Where the list contains multiple dictionaries, and each dictionary corresponds to a data sample. We suggest using more than 1000 data samples to run the actor training.
  </details>

* <details><summary> (⚠️WIP) Create the full dataset augmenting few custom data samples </summary>

  The dataset can be generated synthetically from a few prompt+response examples provided by the user (few =>10).
  </details>
</details>

<details>
<summary> Dataset for RLHF </summary>
    
The dataset for RLHF consists just of prompt examples:

```json
[
    {
        "user_input": "here the example of user input"
    }
]
```

It can be provided in 2 different ways:

* <details><summary> Few examples provided by the user and dataset synthetically expanded using LLM </summary>

    You need to add the key `rlhf` to the `templates.json` file with the information about the task you want to perform and extra context needed by the LLM for the generation. Here is an example of template:

    ```json
    {
      "rlhf": "Here is the template for the generating RLHF prompts. The task we want to perform is ..."
    }
    ```

     *Note that all templates must be saved in a single JSON file named `templates.json`*
     </details>

* <details><summary> The user provides the full dataset with possible interactions with the model </summary>

    The dataset needs to contain more than 1000 prompt examples:

    ```json
    [
        {
            "user_input": "here the example of user input"
        }
    ]
    ```

    The file must be named `rlhf_training_data.json`.
    </details>
</details>
<details>
<summary><b> Dataset to train the reward model </b></summary>

The `reward_training_data` is a collection of i) prompts, ii) completion and iii) score of the completion assigned accordingly to the user feedback (the Human Feedback in RLHF). 

```json
[{
	"user_input": "...",
	"completion": "...",
	"score": 1
},
	...
]
```

We support 3 different options to prepare the `reward_training_data`: 

- Fully Synthetic Score Generation
    
    In this case the reward dataset can be synthetically scored using a LLM as Human Feedback. We recommend the `reward_training_data` having at least 100 data samples.
    
    ```json
    [{
    	"user_input": "...",
    	"completion": "...",
    	"score": None
    },
    	...
    ]
    ```
    
    A LLM model is used to assign the score to each entry. 
    
    The LLM needs a prompt template containing all the instructions to evaluate the generated text. To do this, you should add the key `reward` to the `templates.json` file. Here is an example:
    
    ```json
    {
    	"reward": "Here is the template for the reward model. The rules are:\n\n1.Rule 1\n\n2. Rule 2"
    }
    ```
    
    If no template is provided the default one is used. You can find the default template in `artifacts/generate_rewards.py`. Note that all templates must be saved in a single JSON file named `templates.json`. 
    
    Once you have the unlabelled dataset, you can generate the scores by running the following command:
    
    ```bash
    python artifacts/generate_rewards.py <dataset_path> --model <model_to_use> --temperature <t> --max_tokens <n> --reward_template <path_to_file.json>
    ```
    
    Where:
    
    - `<dataset_path>` path to the reward dataset to be scored;
    - `<model_to_use>` model to use for the reward. Default and suggested text-davinci-003 (More to come);
    - `<temperature>` temperature used to score the model; temperature=0.1;
    - `<max_tokens>` max_tokens of the generation;
    - `<reward_template>` is the path to the `templates.json` file containing the template to be used for generating the reward. If no path is provided, the default template will be used.
- The user provides their personalized full dataset
    
    Datasets must be JSON files in the following format:
    
    ```json
    [
        {
            "user_input": "here type the user input",
            "completion": "here type the completion",
            "score": 4.0
        },
        {
            "user_input": "here type the user input",
            "completion": "random garbage",
            "score": 0.0
        }
    ]
    ```
    
    Note that at least 100 data samples are required in this case. The file must be named `reward_training_data.json`
    
- **(⚠️WIP)** Few examples provided by the user and dataset synthetically expanded using LLM
</details>

# License

See the [LICENSE](https://github.com/nebuly-ai/nebullvm/blob/main/apps/accelerate/chatllama/LICENSE) file.
