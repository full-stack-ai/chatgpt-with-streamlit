import json
from typing import Dict
import random
from contextlib import contextmanager, redirect_stdout
from io import StringIO

def save_chat_history(message_history: Dict) -> None:
    rand_num = random.randint(0,1000000)
    with open(f"messages_{rand_num}.json","w") as file:
        json.dump(message_history, file, indent=4)


@contextmanager
def st_stream_from_stdout(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write
        yield