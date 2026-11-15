
import os
import json
import time

from openai import OpenAI
from dotenv import load_dotenv
from types import SimpleNamespace

from utils.util import log, error, warn
from utils.tools import llm_tools, run_tool

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
        model=model,
        messages=history,
        tools=llm_tools,
        stream=True
    )

    response = ""
    tool_calls = {}

    print("Assistant: ", end="", flush=True)

    for chunk in stream:

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        # Stream assistant content
        if delta.content:
            response += delta.content
            print(delta.content, end="", flush=True)

        # Merge streamed tool calls
        for tc in (delta.tool_calls or []):

            index = tc.index

            if index not in tool_calls:
                tool_calls[index] = SimpleNamespace(
                    id="",
                    type="function",
                    function=SimpleNamespace(
                        name="",
                        arguments=""
                    )
                )

            obj = tool_calls[index]

            if tc.id:
                obj.id = tc.id

            if tc.function:

                if tc.function.name:
                    obj.function.name += tc.function.name

                if tc.function.arguments:
                    obj.function.arguments += tc.function.arguments

    print()

    calls_list = [
        tool_calls[index]
        for index in sorted(tool_calls)
    ]

    return response, calls_list


def build_tool_calls_list(tools):

    payload = []

    for tool in tools:

        payload.append({
            "id": tool.id,
            "type": "function",
            "function": {
                "name": tool.function.name,
                "arguments": tool.function.arguments
            }
        })

    return payload


def execute_tool_calls(tool_calls):

    for tool in tool_calls:

        try:
            arguments = json.loads(
                tool.function.arguments
            )

            log(f"Executing tool: {tool.function.name}")

            # Adapt run_tool to accept parsed arguments
            data = run_tool(tool)

            messages.append({
                "role": "tool",
                "tool_call_id": tool.id,
                "content": json.dumps(data, default=str)
            })


        except Exception as e:

            error(f"Tool execution failed: {e}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool.id,
                "content": json.dumps({
                    "error": str(e)
                })
            })


while True:

    prompt = input("\nUser: ")

    if prompt.lower() in ["exit", "bye"]:
        break

    messages.append({
        "role": "user",
        "content": prompt
    })

    for step in range(MAX_ROUNDS):

        response, tool_calls = stream_chat(messages)

        # No tools requested: conversation turn completed
        if not tool_calls:

            messages.append({
                "role": "assistant",
                "content": response
            })

            break

        # Add assistant tool-call message to history
        messages.append({
            "role": "assistant",
            "content": response or None,
            "tool_calls": build_tool_calls_list(tool_calls)
        })

        log(
            f"Step {step + 1}: "
            f"Model requested {len(tool_calls)} tool call(s)"
        )

        execute_tool_calls(tool_calls)

    else:

        fallback = "Sorry, agent ran out of time."

        messages.append({
            "role": "assistant",
            "content": fallback
        })

        print(f"Qwen: {fallback}")

