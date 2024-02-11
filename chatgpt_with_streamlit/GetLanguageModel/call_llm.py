from dotenv import find_dotenv, load_dotenv
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import LLMChain
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI

system_message = """ You are a respectful AI assistant and your role is to address user's
question to the best of your knowledge. You follow up with the human if they need help with
anything else. Provide your best answer to user's question.
"""

human_template = "{question}"

messages = [
    SystemMessagePromptTemplate.from_template(system_message),
    HumanMessagePromptTemplate.from_template(human_template)
]

chat_prompt = ChatPromptTemplate.from_messages(messages)

_ = load_dotenv(find_dotenv())


def setup_llm_callback_manager():
    handler = StreamingStdOutCallbackHandler()
    callback_manager = CallbackManager([handler])
    return callback_manager


def setup_llm_openai_obj(callback_manager):
    llm = ChatOpenAI(
        callback_manager=callback_manager,
        streaming=True,
        tags=["my-chatgpt"],
        temperature=0
    )
    return llm


def setup_llm_local_obj(callback_manager):
    llm = LlamaCpp(
        model_path="./models/beagle-7b.gguf",
        temperature=0,
        max_tokens=1000,
        top_p=1,
        n_ctx=2048,
        verbose=True,  # Verbose is required to pass to the callback manager
        f16_kv=True,
        callback_manager=callback_manager,
        tags=["local-llm"]
    )
    return llm


def setup_llm_chain(llm):
    llm_chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
    ).with_config({
        "run_name": "llm-chat"
    })
    return llm_chain


def get_chain(llm_option="openai"):
    callback_manager = setup_llm_callback_manager()
    if llm_option == "openai":
        llm = setup_llm_openai_obj(callback_manager)
    else:
        llm = setup_llm_local_obj(callback_manager)

    chain = setup_llm_chain(llm)

    return chain
