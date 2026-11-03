import os

from dotenv import load_dotenv
from openai import OpenAI
from types import SimpleNamespace

from utils.tools import llm_tools

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

prompt = input("user: ")

stream = client.chat.completions.create(
    model=os.getenv("MODEL"),
    stream=True,
    tools=llm_tools,
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt}
    ]
)

print("assistant: ")

tc_history = {}

for chunk in stream:
    if chunk.choices[0].delta.tool_calls:
        print("done")
        for tc in chunk.choices[0].delta.tool_calls:
            tc_obj = tc_history.setdefault(
                tc.id, 
                SimpleNamespace(id=None, function=SimpleNamespace(name="", arguments="")),
            )
            
            if tc.index:
                tc_obj.index = tc.index

            if tc.function:
                if tc.function.name:
                    tc_obj.function.name += tc.function.name
                if tc.function.arguments:
                    tc_obj.function.arguments += tc.function.arguments

            tc_history[tc.id] = tc_obj
        
    print(chunk.choices[0].delta.tool_calls)

# print(tc_history)
for (key) in sorted(tc_history):
    print(tc_history[key])


