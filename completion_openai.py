import openai


MODEL = "gpt-4"
TEMPERATURE = 0.2  # Recommended for deterministic and focused output that is 'more likely to be correct and efficient'.

with open('openai.key', 'r') as file:
    openai.api_key = file.read().rstrip()

def create_chat_completion(prompts, model=MODEL, temperature=TEMPERATURE, **kwargs):
    response = openai.ChatCompletion.create(messages=prompts, model=model, temperature=temperature, **kwargs)
    return response["choices"][0]["message"]["content"]


