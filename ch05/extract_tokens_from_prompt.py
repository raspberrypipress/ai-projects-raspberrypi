from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
messages = [
    {"role": "user", "content": "explain quantum computing to me like I'm a five year old."}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
)
print("Input IDs:", input_ids["input_ids"])
decoded_tokens = [tokenizer.decode([token_id]) for token_id in input_ids["input_ids"]]

# Print them one by one with their id
i = 0
for id, token in zip(input_ids["input_ids"], decoded_tokens):
    print(f"Token #{i:02d} ID {id:>7}: {repr(token)}")
    i += 1


# ollama prompt from the ollama log
"""
prompt="<|start_header_id|>user<|end_header_id|>\n\nexplain quantum computing to me like I'm a five year old<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
"""