# test_api.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
key = os.getenv("OPENROUTER_API_KEY")
print("Key:", key)

client = OpenAI(
    api_key=key,
    base_url="https://openrouter.ai/api/v1"
)

resp = client.chat.completions.create(
    model="google/gemma-3-12b-it:free",
    messages=[{"role": "user", "content": "hi"}]
)
print(resp.choices[0].message.content)