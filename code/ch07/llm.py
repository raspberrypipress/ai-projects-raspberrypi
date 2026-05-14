from hailo_platform import VDevice
from hailo_platform.genai import LLM
from rich.console import Console
import sys
import signal

class LLMApp:

    def __init__(self, model_path, console, prompt):
        self.console = console

        self.console.print("Initialising device...")
        params = VDevice.create_params()
        params.group_id = "SHARED"
        self.vdevice = VDevice(params)

        self.console.print(f"Loading model {model_path}...")
        self.llm = LLM(self.vdevice, model_path)

        # Add system prompt to the LLM's context.
        self.console.print("Initialising model...")
        self.sysmsg = {"role": "system", "content": prompt}
        self.initialise_context()

    def initialise_context(self):
        self.llm.clear_context()
        # Only a single token is needed to add to the context.
        for token in self.llm.generate(prompt=[self.sysmsg],
                                       max_generated_tokens=1):
            pass

    def generate(self, user_input, 
                 max_tokens=200, temperature=0.2):
        # Clear context if we're getting close to the limit.
        capacity = self.llm.max_context_capacity()
        usage = self.llm.get_context_usage_size()
        if usage / capacity > 0.90:
            self.console.print("Context full, clearing.")
            self.initialise_context()

        msg = {"role": "user", "content": user_input}
        r = ""
        try:
            with self.console.status("[blue]Generating..."):
                for token in self.llm.generate(
                            prompt=[msg],
                            temperature=temperature,
                            max_generated_tokens=max_tokens):
                    r += token

        except Exception as e:
            self.console.log(f"Error occurred: {repr(e)}")
            sys.exit(1)

        # Remove end-of-message tokens.
        return r.replace("<|im_end|>", "")

    def __del__(self):
        self.llm.clear_context()
        self.llm.release()
        self.vdevice.release()


if __name__ == "__main__":
    # Set up some rich consoles for diags and output.
    diags = Console(stderr=True, style="purple")
    output = Console(style="dark_cyan")

    app = LLMApp("./Qwen2-1.5B-Instruct.hef", diags,
                 "You are a helpful assistant.")

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    diags.print("Ready.")
    for text in sys.stdin:
        response = app.generate(text.strip())
        output.print(response)

    diags.print("Farewell from llm.py")