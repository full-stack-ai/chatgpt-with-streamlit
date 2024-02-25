from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts.chat import (AIMessagePromptTemplate,
                                    ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
# Load environment variables
_ = load_dotenv(find_dotenv()) # Remember to run $unset OPENAI_API_KEY from your terminal after you're done!

# Setup LLM Objects

# Setup callback manager to enable streaming to stdout
def setup_llm_callback_manager():
    handler = StreamingStdOutCallbackHandler()
    callback_manager = CallbackManager([handler])
    return callback_manager


# Handling Openai API
def setup_llm_openai_obj(callback_manager=None):
    llm = ChatOpenAI(
        callback_manager=callback_manager,
        streaming=True,
        temperature=0
        # tags for LangSmith
    )
    return llm

# Handling Local LLMs
def setup_llm_local_obj(llm_name, callback_manager=None):
    llm_pathname = "./models/" + llm_name
    llm = LlamaCpp(
        model_path = llm_pathname,
        temperature=0,
        max_tokens=150,
        top_p=1,
        n_ctx=2048,
        verbose=True,
        f16_kv=True,
        callback_manager=callback_manager,
        # tags for langsmith
    )
    return llm

# Setup LLM Chain
def setup_llm_chain(llm, prompt):
    chain = LLMChain(
        llm=llm,
        prompt=prompt
    ) # Add run_name for LangSmith Configs
    return chain

# Create LLM Chain

# Chat LLMChain
def get_chat_llm_chain(llm_option="openai") -> LLMChain:
    sys_template = """
            You are a respectful AI assistant and your role is to address user's
            question to the best of your knowledge. You follow up with the human if they need help with
            anything else. Provide your best answer to user's question.
            """
    
    human_template = "{question}"

    messages = [
        SystemMessagePromptTemplate.from_template(sys_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ]

    chat_prompt = ChatPromptTemplate.from_messages(messages=messages)
    # Setup callback manager
    callback_manager = setup_llm_callback_manager()

    if llm_option == "openai":
        llm = setup_llm_openai_obj(callback_manager=callback_manager)
    else:
        llm = setup_llm_local_obj(llm_name=llm_option, callback_manager=callback_manager)
    
    chain = setup_llm_chain(llm, chat_prompt)

    return chain

# Summary Chain
def get_summary_llm_chain() -> LLMChain:
    sys_template = """ Find the user's question in the messages and describe it in 4 words
    """

    human_template = "{messages}"

    ai_template = "Tope 5 words are: \n"

    messages = [
        SystemMessagePromptTemplate.from_template(sys_template),
        HumanMessagePromptTemplate.from_template(human_template),
        AIMessagePromptTemplate.from_template(ai_template)
    ]

    summary_prompt = ChatPromptTemplate.from_messages(messages=messages)

    llm = setup_llm_openai_obj() # no need to pass any callback manager

    summary_chain = setup_llm_chain(llm=llm, prompt=summary_prompt)

    return summary_chain


