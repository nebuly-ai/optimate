from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains.conversation.memory import (
    ConversationBufferWindowMemory,
)

from chatllama.langchain_modules.prompt_templates import (
    PERSON_CHATBOT_TEMPLATE,
    AI_CHATBOT_TEMPLATE,
)


CONVERSATION_LENGTH = 20


def create_conversation(human_agent: LLMChain, bot_agent: LLMChain):
    conversation = []
    chatbot_output = ""
    for i in range(CONVERSATION_LENGTH):
        # Human agent goes first
        human_output = human_agent.run(chatbot_input=chatbot_output)
        conversation.append(f"Human: {human_output}")
        chatbot_output = bot_agent.run(human_input=human_output)
        conversation.append(f"AI: {chatbot_output}")
    return "\n".join(conversation)


def build_agents():
    # be aware that too long completions will not fit the sequence length
    # of possible critic or reward models ...
    llm = OpenAI(max_tokens=2048, temperature=0.7)
    human_template = PromptTemplate(**PERSON_CHATBOT_TEMPLATE)
    human_agent = LLMChain(
        llm=llm,
        prompt=human_template,
        memory=ConversationBufferWindowMemory(k=4),
    )
    bot_template = PromptTemplate(**AI_CHATBOT_TEMPLATE)
    bot_agent = LLMChain(
        llm=llm,
        prompt=bot_template,
        memory=ConversationBufferWindowMemory(k=4),
    )
    return human_agent, bot_agent


def get_sub_conversations(conversation: str, system_prompt: str):
    interactions = conversation.split("AI:")
    sub_conversations = []
    for i in range(len(interactions) - 1):
        user_input = system_prompt + "AI:".join(interactions[: i + 1])
        completion = interactions[i + 1].split("Human:")[0].strip()
        sub_conversations.append(
            {"user_input": user_input, "completion": completion}
        )
    return sub_conversations


def main():
    import json
    import os
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--num_conversations", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="conversations")
    parser.add_argument("--templates", type=str, default=None)
    args = parser.parse_args()

    if args.templates is not None:
        with open(args.templates, "r") as f:
            templates = json.load(f)
        template = templates["actor"]
    else:
        template = ""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for conv in range(args.num_conversations):
        human_agent, bot_agent = build_agents()
        conversation = create_conversation(human_agent, bot_agent)
        with open(
            os.path.join(args.output_dir, f"conversation_{conv}.txt"), "w"
        ) as f:
            f.write(conversation)

    # convert the conversations to a single json file
    data = []
    for conv in range(args.num_conversations):
        with open(
            os.path.join(args.output_dir, f"conversation_{conv}.txt"), "r"
        ) as f:
            conversation = f.read()
        sub_conversations = get_sub_conversations(conversation, template)
        data.extend(sub_conversations)
    with open(
        os.path.join(args.output_dir, "actor_training_data.json"), "w"
    ) as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
