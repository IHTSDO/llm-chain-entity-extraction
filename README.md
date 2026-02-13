# Using AI Chains for SNOMED CT entity extraction from free text clinical notes

LLM Chains allow us to divide a complex task into a Chain of smaller tasks, where the output of one prompt becomes the input of the next.

This is not significantly slower, despite requiring a greater number of calls/requests to the LLM, because short outputs take a proportionally shorter time to generate. Although the input prompt tokens also need to be processed each time, they are less time-expensive than output generated tokens in the case of LLaMA 2.

## Chain design
Note: the diagram below is currently outdated and will be refreshed in a future update.
<img width="800" alt="image" src="https://github.com/IHTSDO/llm-chain-entity-extraction/assets/4990842/bc520330-62d3-4bb6-8d37-6638df721587">


## How to run the Entity Extractor
The `entity_extractor.py` script extracts clinical entities from free text using LLMs and maps them to SNOMED CT.
It supports `openai`, `llama`, and `bard` as API backends.

For OpenAI, the project is configured for the current Python SDK and uses the modern Responses API (with a compatibility fallback), including support for ChatGPT-5 family models (for example `gpt-5-mini`).

### Prerequisites
To run `entity_extractor.py`, you will need:
- Python 3.x
- OpenAI: an API key for OpenAI API (see https://platform.openai.com/)
- Llama-2 (optional): `llama.cpp` installed, and a local copy of Llama-2 quantized for use in `llama.cpp` (see https://github.com/ggerganov/llama.cpp)
  
### Usage
Follow these steps:

- Clone the repository to your local machine.
- Open the project in your preferred code editor.
- Install dependencies: `pip install -r requirements.txt`
- If you need to use OpenAI models, create a local `openai.key` file in the project root and paste your API key there.
- Run the script with:

```bash
python entity_extractor.py --api <api> --model <model> --sentences <path>
```

### CLI parameters

- `-a, --api`: backend to use (`openai`, `llama`, `bard`). Default: `openai`.
- `--model`: model identifier.
  - OpenAI example: `gpt-5-mini`
  - Llama example: local model file path
- `--sentences`: path to a text file containing one clinical sentence per line.
- `--n`: optional limit for number of non-empty, non-comment lines to process.
- `--mode`: output mode.
  - `verbose` (default): shows matching stages and detailed output.
  - `summary`: shows only `Processing sentence X...`, then final colored sentence output and per-line score.

### Optional environment variables

- `OPENAI_DEBUG_CACHE=1`: enables request/response cache debug logs from the OpenAI client layer.

Examples:

```bash
python3.10 entity_extractor.py --api llama --model /my-drive/models/llama2/llama-2-13b-chat/ggml-model-q4_0.bin --sentences=example_cases/clinical_text.txt

python3 entity_extractor.py --api openai --model gpt-5-mini --sentences=example_cases/clinical_text.txt

OPENAI_DEBUG_CACHE=1 python3 entity_extractor.py --api openai --model gpt-5-mini --sentences=example_cases/clinical_text.txt --mode summary
```

## References

- Wu, T., Terry, M., & Cai, C. J. (2022, April). AI chains: Transparent and controllable human-ai interaction by chaining large language model prompts. In Proceedings of the 2022 CHI conference on human factors in computing systems (pp. 1-22).
https://doi.org/10.1145/3491102.3517582 
- Cursor https://www.cursor.so/blog/llama-inference#user-content-fn-llama-paper [Retrieved 08/2023]
- LLaMA 2 – Meta AI https://ai.meta.com/llama/ [Retrieved 08/2023]
