from hailo_platform import VDevice
from hailo_platform.genai import LLM
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils \
    import (
        message_formatter,
        context_manager,
        agent_utils,
    )
from hailo_apps.python.core.common.defines \
    import SHARED_VDEVICE_GROUP_ID
from pathlib import Path
import sys
import signal

signal.signal(signal.SIGINT, signal.SIG_IGN)
model_name = "Qwen3-1.7B-Instruct"

hef_path = Path.home() / "Downloads" / f"{model_name}.hef"
print(f"Using model {hef_path}", file=sys.stderr)

vdevice = None
llm = None
try:
    params = VDevice.create_params()
    params.group_id = SHARED_VDEVICE_GROUP_ID
    vdevice = VDevice(params)
    print("Hailo device initialized. Loading model...", 
            file=sys.stderr)

    llm = LLM(vdevice, str(hef_path))
    print("Model loaded successfully", file=sys.stderr )

    prompt = "You are a helpful assistant."
    messages = [message_formatter.messages_system(prompt)]
    for text in sys.stdin:
        msg = message_formatter.messages_user(text.strip())
        print(f"User input: {text}", file=sys.stderr)
        messages.append(msg)

        r = llm.generate_all(prompt=messages, 
                             temperature=0.1,
                             seed=42,
                             max_generated_tokens=512)

        r = r.split(". [{'type'")[0]
        r = r.replace("<|im_end|>", "")
        print(r, file=sys.stderr)
        response = message_formatter.messages_assistant(r)
        messages.append(response)
        context_manager.print_context_usage(llm)
    print("Farewell from llm.py", file=sys.stderr)

except Exception as e:
    print(f"Error occurred: {e}", file=sys.stderr)
    sys.exit(1)

finally:
    agent_utils.cleanup_resources(llm, vdevice)