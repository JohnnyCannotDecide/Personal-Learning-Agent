from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    model="google/gemma-3-12b-it:free",
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
)
print(llm.invoke("hi").content)