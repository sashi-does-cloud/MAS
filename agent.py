# Standard libraries
import os, json, time

# Third-party packages
from openai import OpenAI 
from dotenv import load_dotenv
from types import SimpleNamespace

# Custom packages
from utils.util import log,error,warn
from utils.tools import llm_tools, get_current_time, name_to_fn, run_tool

load_dotenv()

base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")
model = os.getenv("MODEL")


client = OpenAI(
    base_url=base_url,
    api_key=api_key
)


messages = []

MAX_ROUNDS = 5

def stream_chat(history):

    stream = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=history,
        tools=llm_tools,
        stream=True
    )

    response = ""
    tool_calls = {}

    for chunk in stream:

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta 
        init = False

        # merge content
        if delta.content:
            if not init:
                print("Assistant: ", end=True, flush=True)
                init = True
            response += delta.content

        # merge tool calls
        for tc in (delta.tool_calls or []):

            obj = calls.setdefault(
                tc.id,
                SimpleNamespace(id=None, function=SimpleNamespace(name="", arguments=""))
            )

            if tc.id:
                obj.index = tc.index
            
            if tc.function:
                if tc.function.name:
                    obj.function.name += tc.function.name 
                if tc.function.arguments:
                    obj.function.arguments += tc.function.arguments
        
        if init:
            print()

        calls_list = []
        for (key) in sorted(calls):
            calls_list.append(calls[key])
        
        return response, calls_list


def build_tool_calls_list(tools):
    payload = []
    for tool in tool_calls:
        payload.append({
            "id": tool.id,
            "type": "function",
            "function": {
                "name": tool.function.name,
                "arguments": tool.function.arguments
            }
        })
    
    return payload

while True:

    prompt = input("\nuser: ")
    if prompt.lower() in ["exit", "bye"]:
        break
    
    messages.append({
        "role": "user",
        "content": prompt
    })

    for step in range(MAX_ROUNDS):

        response, tool_calls = stream_chat(messages)

        # response = client.chat.completions.create(
        #     model=model,
        #     messages=messages,
        #     tools=llm_tools,
        # )  

        if not tool_calls:
            break

        messages.append({
            "role": "assistant", 
            "content": response,
            "tool_calls": build_tool_calls_list(tool_calls) 
        })

        log(f"\nStep: {step + 1} model requested {len(tool_calls)} tool call(s)\n")

        for tool in tool_calls:

            data = run_tool(tool)

            messages.append({
                "role": "tool",
                "tool_call_id": tool.id,
                "content": str(data)
            })
    else:

        fallback = "Sorry, agent ran out of time"
        messages.append({
            "role": 'assistant',
            "content": fallback
        })

        print("qwen: " + fallback)

        continue
    
    print("qwen: ", end="")
    for chunk in response.choices[0].message.content or "":
        print(chunk, end="", flush=True)
        time.sleep(0.01)
    print("\n")



