from contextlib import contextmanager, redirect_stdout
from io import StringIO
from typing import Dict, Text
from chatgpt_with_streamlit.GetLanguageModel.call_llm import get_summary_llm_chain
import json
# A function call to redirect stdout to streamlit
@contextmanager
def stdout_streaming(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield

# Stringify chat messages
def stringify_message_content(messages: Dict) -> Text:
    message_output = ""
    for message in messages:
        message_string = [message["role"], ": ", message["content"], "\n"]
        message_output += message_output.join(message_string)
    return message_output


# Save message history
def save_chat_history(messages: Dict) -> None:
    chat_string = stringify_message_content(messages)
    summary_llm_chain = get_summary_llm_chain()
    chat_topic = summary_llm_chain.invoke(chat_string)
    with open(f"{chat_topic['text']}_message.json", "w") as file:
        json.dump(messages, file, indent=4)

# Load message history
def load_chat_history(file: Text):
    with open(file, "r") as file:
        message_dict = json.load(file)
    return message_dict

# Remove chat file extension
def remove_file_extension(filename: str):
    extension = "_message.json"
    if filename.endswith(extension):
        return filename[:-len(extension)]
    return filename