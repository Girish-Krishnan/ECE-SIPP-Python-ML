from llama_cpp import Llama

llm = Llama(
    model_path="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4,
    temperature=0.2,
    top_p=0.9,
    repeat_penalty=1.15,
    verbose=False
)

while True:
    user_in = input("You: ").strip()
    if not user_in:
        break

    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You answer briefly and correctly."},
            {"role": "user", "content": user_in}
        ],
        max_tokens=64,
        stop=["</s>"]
    )
    answer = resp["choices"][0]["message"]["content"].strip()
    print("Assistant:", answer)
