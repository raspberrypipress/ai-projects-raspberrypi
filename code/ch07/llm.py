from hailo_platform import VDevice
from hailo_platform.genai import LLM
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils \
    import (
        message_formatter,
        agent_utils,
    )
from hailo_apps.python.core.common.defines \
    import SHARED_VDEVICE_GROUP_ID
from pathlib import Path
import sys
import signal
from rich.console import Console

console = Console(stderr=True)
# model_name = "Qwen3-1.7B-Instruct"
model_name = "Qwen2-1.5B-Instruct"

hef_path = Path.home() / "Downloads" / f"{model_name}.hef"
console.print(f"Using model {hef_path}")

vdevice = None
llm = None
try:
    params = VDevice.create_params()
    params.group_id = SHARED_VDEVICE_GROUP_ID
    vdevice = VDevice(params)
    console.print("Loading model...")

    llm = LLM(vdevice, str(hef_path))
    console.print("Model loaded.")

    prompt = "You are a helpful assistant."
    messages = [message_formatter.messages_system(prompt)]
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for text in sys.stdin:
        msg = message_formatter.messages_user(text.strip())
        messages.append(msg)

        r = ""
        with llm.generate(prompt=messages, 
                          temperature=0.1,
                          seed=42,
                          max_generated_tokens=1024) as gen:
            with console.status("[bold green]Thinking..."):
                for token in gen:
                    r += token

        r = r.split(". [{'type'")[0]
        r = r.replace("<|im_end|>", "")
        print(r)
        response = message_formatter.messages_assistant(r)
        messages.append(response)
        ctx_max = llm.max_context_capacity()
        ctx_used = llm.get_context_usage_size()
        console.print(f"Context usage: {ctx_used}/{ctx_max}")
    console.print("Farewell from llm.py")

except Exception as e:
    console.log(f"Error occurred: {e}")
    sys.exit(1)

finally:
    agent_utils.cleanup_resources(llm, vdevice)