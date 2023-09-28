import multiprocessing
from llama_cpp import Llama


MODEL_PATH = "[PATH TO MODEL]"
# Avoid threading onto efficiency cores. Set to None for automatic.
N_THREADS = max(1, multiprocessing.cpu_count() - 4)
USE_GPU = True
TEMPERATURE = 0
MAX_TOKENS = 2048
llm = None

def initialize_model(model_path):
    global llm
    if model_path:
        MODEL_PATH = model_path
    if USE_GPU:
        print("Using GPU")
        llm = Llama(model_path=MODEL_PATH, n_ctx=MAX_TOKENS, verbose=False, n_gpu_layers=128,
                n_threads=N_THREADS, use_mlock=True)
    else:
        print("Using CPU")
        llm = Llama(model_path=MODEL_PATH, n_ctx=MAX_TOKENS, verbose=False,
                n_threads=N_THREADS, use_mlock=True)

def create_chat_completion(prompts, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, **kwargs):
    response = llm.create_chat_completion(prompts, max_tokens=max_tokens, temperature=temperature, **kwargs)
    return response["choices"][0]["message"]["content"]
