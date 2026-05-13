from hailo_platform import VDevice
from hailo_platform.genai import LLM
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
        params.group_id = "SHARED"
        self.vdevice = VDevice(params)

        self.console.print("Loading model...")
        self.llm = LLM(self.vdevice, str(hef_path))

        # Add system prompt to the LLM's context.
        self.console.print("Initialising model...")
        self.sysmsg = {"role": "system", "content": prompt}
        self.initialize_context()

    def initialize_context(self):
        self.llm.clear_context()
        for token in self.llm.generate(prompt=[self.sysmsg],
                                       max_generated_tokens=1):
            pass

    def generate(self, user_input):
        capacity = self.llm.max_context_capacity()
        usage = self.llm.get_context_usage_size()

        if usage / capacity > 0.90:
            self.console.print("Context full, clearing.")
            self.initialize_context()

        msg = {"role": "user", "content": user_input}
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
        self.llm.clear_context()
        self.llm.release()
        self.vdevice.release()

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