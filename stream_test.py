import os, time

from openai import OpenAI
from dotenv import load_dotenv

from utils.tools import llm_tools


load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

messages = [
    {
        "role": "user",
        "content": "what is the weather at tirupati"
    }
]

stream = client.chat.completions.create(
    model=os.getenv("MODEL"),
    messages=messages,
    tools=llm_tools,
    stream=True
)


for chunk in stream:

    if chunk.choices is None or chunk.choices[0].delta.content is None:
        continue

    for char in chunk.choices[0].delta.tool_calls:
        print(char, end="", flush=True)
        time.sleep(0.01)


