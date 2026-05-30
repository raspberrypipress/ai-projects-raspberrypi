from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
messages = [
    {"role": "user", "content": "explain quantum computing to me like I'm a five year old."}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=False,
    return_tensors="pt"
)
decoded_tokens = [tokenizer.decode([token_id]) for token_id in input_ids["input_ids"][0]]

# Print them one by one with their id
i = 1
for id, token in zip(input_ids["input_ids"][0], decoded_tokens):
    print(f"Token #{i:02d} ID {id:<6}: {repr(token)}")
    i += 1
