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

diagnostics = Console(stderr=True, style="deep_pink4")
output = Console(style="cornflower_blue")

model_name = "Qwen2-1.5B-Instruct"
hef_path = Path.home() / "Downloads" / f"{model_name}.hef"
diagnostics.print(f"Using model {hef_path}")

diagnostics.print("Initialising device...")
params = VDevice.create_params()
params.group_id = SHARED_VDEVICE_GROUP_ID
vdevice = VDevice(params)

diagnostics.print("Loading model...")
llm = LLM(vdevice, str(hef_path))

# Add system prompt to the LLM's context.
diagnostics.print("Initialising model...")
sys_prompt = "You are a helpful assistant."
sys_message = message_formatter.messages_system(sys_prompt)
context_manager.add_to_context(llm, [sys_message])

signal.signal(signal.SIGINT, signal.SIG_IGN)
diagnostics.print("Ready.")
try:
    for text in sys.stdin:

        if context_manager.is_context_full(llm):
            diagnostics.print("Warning: Context full, clearing.")
            llm.clear_context()
            context_manager.add_to_context(llm, [sys_message])

        msg = message_formatter.messages_user(text.strip())

        # Generate the response.
        r = ""
        with llm.generate(prompt=[msg], 
                          temperature=0.8) as gen:
            with diagnostics.status("[green]Thinking...[/green]"):
                for token in gen:
                    r += token

        # Clean response and print it.
        r = streaming.clean_response(r)
        output.print(r)

    diagnostics.print("Farewell from llm.py")

except Exception as e:
    diagnostics.log(f"Error occurred: {e}")
    sys.exit(1)

finally:
    agent_utils.cleanup_resources(llm, vdevice)