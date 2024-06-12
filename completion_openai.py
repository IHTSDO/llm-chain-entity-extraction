import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_HOST"),
)

MODEL = os.environ.get("OPENAI_MODEL") or "gpt-3.5-turbo"
TEMPERATURE = 0
# Temperature 0.2 is recommended for deterministic and focused output that is
# 'more likely to be correct and efficient'. Instead, 0 is used for consistency.

def create_chat_completion(prompts, model=MODEL, temperature=TEMPERATURE, **kwargs):
    response = client.chat.completions.create(messages=prompts, model=model, temperature=temperature, **kwargs)
    return response.choices[0].message.content


