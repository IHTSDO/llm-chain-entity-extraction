import langchain

from langchain.llms import LlamaCpp
from langchain import LLMChain, PromptTemplate

llm = LlamaCpp(model_path="/Users/yoga/llama-cpp/models/llama-2-13b-chat/ggml-model-q4_0.bin")

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_chain.run(question)

