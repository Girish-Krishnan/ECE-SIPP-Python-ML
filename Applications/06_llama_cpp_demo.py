from llama_cpp import Llama


def main():
    # Path to a local GGML or GGUF model file
    llm = Llama(model_path='llama-2-7b.ggmlv3.q4_0.bin')

    while True:
        prompt = input('You: ')
        if not prompt:
            break
        output = llm(prompt, max_tokens=100, stop=['\n', 'User:'])
        print('Assistant:', output['choices'][0]['text'].strip())


if __name__ == '__main__':
    main()
