from hailo_platform import VDevice
from hailo_platform.genai import LLM
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils \
    import (
        message_formatter,
        streaming,
        agent_utils,
        context_manager,
    )
from hailo_apps.python.core.common.defines \
    import SHARED_VDEVICE_GROUP_ID
from rich.console import Console
from pathlib import Path
import sys
import signal

class LLMApp:

    def __init__(self, model, console, prompt):

        self.console = console

        hef_path = Path.home() / "Downloads" / f"{model}.hef"
        self.console.print(f"Using model {hef_path}")

        self.console.print("Initialising device...")
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        self.vdevice = VDevice(params)

        self.console.print("Loading model...")
        self.llm = LLM(self.vdevice, str(hef_path))

        # Add system prompt to the LLM's context.
        self.console.print("Initialising model...")
        sys_message = message_formatter.messages_system(prompt)
        context_manager.add_to_context(self.llm, [sys_message])
        self.sys_message = sys_message

    def generate(self, user_input):

        if context_manager.is_context_full(self.llm, 0.90):
            self.console.print("Context full, clearing.")
            self.llm.clear_context()
            context_manager.add_to_context(self.llm,
                                           [self.sys_message])

        msg = message_formatter.messages_user(user_input)
        r = ""
        try:
            with self.console.status("[blue]Generating..."):
                for token in self.llm.generate(prompt=[msg]):
                    r += token

        except Exception as e:
            self.console.log(f"Error occurred: {repr(e)}")
            sys.exit(1)

        return streaming.clean_response(r)

    def __del__(self):
        agent_utils.cleanup_resources(self.llm, self.vdevice)

if __name__ == "__main__":
    # Set up some rich consoles for diags and output.
    diags = Console(stderr=True, style="purple")
    output = Console(style="dark_cyan")

    app = LLMApp("Qwen2-1.5B-Instruct", diags,
                 "You are a helpful assistant.")

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    diags.print("Ready.")
    for text in sys.stdin:
        response = app.generate(text.strip())
        output.print(response)

    diags.print("Farewell from llm.py")