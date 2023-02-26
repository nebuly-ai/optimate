REWARD_TEMPLATE = dict(
    template=(
        "Lets pretend that you are a lawyer and you have to"
        "evalaute the following completion task from a given"
        "assigment with a score between 0 and 5 where 0 represents"
        "a bad assignment completion and 5 a perfect completion.\n"
        "You MUST evaluate: text quality, content quality and"
        "coherence.\n"
        "You MUST return only the number that represents your"
        "judgment.\n"
        "The assignement is:\n{user_input}\n"
        "The completion is:\n{completion}\n"
    ),
    input_variables=["user_input", "completion"],
)
