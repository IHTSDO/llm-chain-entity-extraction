# Using AI Chains for SNOMED CT entity extraction from free text clinical notes

LLM Chains allow us to divide a complex task into a Chain of smaller tasks, where the output of one prompt becomes the input of the next.

This is not significantly slower, despite requiring a greater number of calls/requests to the LLM, because short outputs take a proportionally shorter time to generate. Although the input prompt tokens also need to be processed each time, they are less time-expensive than output generated tokens in the case of LLaMA 2.

## Chain design
<img width="800" alt="image" src="https://github.com/IHTSDO/llm-chain-entity-extraction/assets/4990842/bc520330-62d3-4bb6-8d37-6638df721587">


## How to run the Entity Extractor
The entity_extractor.py script extracts clinical entities from free text using LLMs. It uses the FHIR API to retrieve clinical data and the OpenAI, BARD, or Llama API to extract entities from the text.

### Prerequisites
To run the entity_extractor.py script, you will need:
- Python 3.x
- OpenAI: An API key for OpenAI Api (see https://openai.com/blog/openai-api)
- Llama-2: Llama.cpp installed, and a local copy of Llama-2, quantized for use in llama.cpp (see https://github.com/ggerganov/llama.cpp)
  
### Usage
To use the entity_extractor.py script, follow these steps:

- Clone the repository to your local machine.
- Open the project in your preferred code editor.
- Install any necessary dependencies by running pip install -r requirements.txt in your terminal.
- If you need to use OpenAI models, save your OpenAI API key in a file named "openai.key".
- Run the script by typing python entity_extractor.py -a <api> --model <model> in your terminal, where <api> is the name of the LLM API you want to use (openai, bard, or llama) and <model> is the name of the model you want to run for OpenAI or path to the model in the case of Llama-2.
- View the output in the terminal or in the output pane of your code editor.

Examples:

```
python3.10 entity_extractor.py --api llama --model /my-drive/models/llama2/llama-2-13b-chat/ggml-model-q4_0.bin --sentences=example_cases/clinical_text.txt

python3 entity_extractor.py --api openai --model gpt-4o  --sentences=example_cases/clinical_text.txt
```

## References

- Wu, T., Terry, M., & Cai, C. J. (2022, April). AI chains: Transparent and controllable human-ai interaction by chaining large language model prompts. In Proceedings of the 2022 CHI conference on human factors in computing systems (pp. 1-22).
https://doi.org/10.1145/3491102.3517582 
- Cursor https://www.cursor.so/blog/llama-inference#user-content-fn-llama-paper [Retrieved 08/2023]
- LLaMA 2 – Meta AI https://ai.meta.com/llama/ [Retrieved 08/2023]

