from hailo_platform import VDevice
from hailo_platform.genai import LLM
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import (
    message_formatter,
    context_manager,
    agent_utils,
)
from hailo_apps.python.core.common.defines import SHARED_VDEVICE_GROUP_ID
from pathlib import Path
import sys

if __name__ == "__main__":

    hef_path = Path.home() / "Downloads" / "Qwen3-1.7B-Instruct.hef"
    print(f"Using model {hef_path}")

    vdevice = None
    llm = None
    try:
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        vdevice = VDevice(params)
        print("Hailo device initialized")

        llm = LLM(vdevice, str(hef_path))
        print("Model loaded successfully")

        messages = [message_formatter.messages_system("You are a helpful assistant.")]
        while True:
            user_input = input("> ")
            if user_input.lower() in ["quit", "exit", "bye"]:
                break
            messages.append(message_formatter.messages_user(user_input))

            response = llm.generate_all(prompt=messages, temperature=0.1, seed=42, max_generated_tokens=512)

            print("-" * 60)
            cleaned_response = response.split(". [{'type'")[0]
            cleaned_response = cleaned_response.replace("<|im_end|>", "")
            print(f"Response: {cleaned_response}")
            messages.append(message_formatter.messages_assistant(cleaned_response))
            context_manager.print_context_usage(llm, show_always=True)
            print(messages)
            print("-" * 60)
        print("Done")

    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

    finally:
        agent_utils.cleanup_resources(llm, vdevice)