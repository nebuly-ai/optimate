from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains.conversation.memory import (
    ConversationalBufferWindowMemory,
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
        memory=ConversationalBufferWindowMemory(k=4),
    )
    bot_template = PromptTemplate(**AI_CHATBOT_TEMPLATE)
    bot_agent = LLMChain(
        llm=llm,
        prompt=bot_template,
        memory=ConversationalBufferWindowMemory(k=4),
    )
    return human_agent, bot_agent


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--num_conversations", type=int, default=1000)
    parser.add_argument("--output_file", type=str, default="conversations.txt")
    args = parser.parse_args()
    conversations = []
    for conv in range(args.num_conversations):
        human_agent, bot_agent = build_agents()
        conversation = create_conversation(human_agent, bot_agent)
        conversations.append(conversation)
    with open(args.output_file, "w") as f:
        f.write("\n\nNEW CONVERSATION\n\n".join(conversations))

if __name__ == "__main__":
    main()