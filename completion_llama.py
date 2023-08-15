import multiprocessing
from llama_cpp import Llama


# MODEL_PATH = "/Users/alo/llm/llama2/llama-2-13b-chat/ggml-model-q4_0.bin"
MODEL_PATH = "/Users/yoga/llama-cpp/models/llama-2-13b-chat/ggml-model-q4_0.bin"
# Avoid threading onto efficiency cores. Set to None for automatic.
N_THREADS = max(1, multiprocessing.cpu_count() - 4)
USE_GPU = True
TEMPERATURE = 0
MAX_TOKENS = 2048

if USE_GPU:
    llm = Llama(model_path=MODEL_PATH, n_ctx=MAX_TOKENS, verbose=False, n_gpu_layers=128,
                n_threads=N_THREADS, use_mlock=True)
else:
    llm = Llama(model_path=MODEL_PATH, n_ctx=MAX_TOKENS, verbose=False,
                n_threads=N_THREADS, use_mlock=True)


def create_chat_completion(prompts, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, **kwargs):
    response = llm.create_chat_completion(prompts, max_tokens=max_tokens, temperature=temperature, **kwargs)
    return response["choices"][0]["message"]["content"]
