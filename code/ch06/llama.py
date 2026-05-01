from llama_cpp import Llama
model_file = "/path/to/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
llm = Llama(model_path=model_file , chat_format="chatml")

chat_history=[
    {
        "role": "system",
        "content": ("You are a wise-cracker who gives "
            "helpful answers with only a few words"),
        }
    ]

while True:
    question = input("What is your question? ")
    if question.lower() == "exit":
        break

    chat_history.append({"role": "user", "content": question})
    output = llm.create_chat_completion(chat_history,
                                        temperature=0.7,
                                        max_tokens=40)
    response = output['choices'][0]['message']['content']
    print(output['choices'][0]['message']['content'])
    if(output['choices'][0]['finish_reason'] == 'length'):
        print("ran out of time ... ")