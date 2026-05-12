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

console = Console(stderr=True)

model_name = "Qwen2-1.5B-Instruct"
hef_path = Path.home() / "Downloads" / f"{model_name}.hef"
console.print(f"Using model {hef_path}")

console.print("Initialising device...")
params = VDevice.create_params()
params.group_id = SHARED_VDEVICE_GROUP_ID
vdevice = VDevice(params)

console.print("Loading model...")
llm = LLM(vdevice, str(hef_path))

sys_prompt = "You are a helpful assistant."
sys_message = message_formatter.messages_system(sys_prompt)
llm.add_to_context(sys_message)

signal.signal(signal.SIGINT, signal.SIG_IGN)
console.print("Ready.")
try:
    for text in sys.stdin:

        if context_manager.is_context_full(llm):
            console.print("Warning: Context full, clearing.")
            llm.clear_context()

        msg = message_formatter.messages_user(text.strip())

        # Generate the response.
        r = ""
        with llm.generate(prompt=[msg], 
                          temperature=0.8) as gen:
            with console.status("[bold green]Thinking..."):
                for token in gen:
                    r += token

        # Clean response and print, then add it to the messages
        r = streaming.clean_response(r)
        print(r)

    console.print("Farewell from llm.py")

except Exception as e:
    console.log(f"Error occurred: {e}")
    sys.exit(1)

finally:
    agent_utils.cleanup_resources(llm, vdevice)