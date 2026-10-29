# Standard libraries
import os, json, time

# Third-party packages
from openai import OpenAI 
from dotenv import load_dotenv

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

# messages.append({
#     "role": "system",
#     "content": "You are an AI assistant with executable tools. "
#             "Understand the user's intent and use the available tools "
#             "whenever they are needed to answer the request. "
#             "Do not merely describe tools or ask the user to perform "
#             "actions that the tools can perform. "
#             "For requests requiring multiple tools, execute all necessary "
#             "tools and use their actual results. If one tool requires "
#             "another tool's output, execute them in the correct order. "
#             "Never invent tool results. Ask for clarification only when "
#             "essential information cannot be obtained from the available "
#             "tools or conversation. After all required tool calls finish, "
#             "provide one clear, concise answer addressing every part "
#             "of the user's request."
# })

while True:

    prompt = input("\nuser: ")
    if prompt.lower() in ["exit", "bye"]:
        break
    
    messages.append({
        "role": "user",
        "content": prompt
    })

    for step in range(MAX_ROUNDS):

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=llm_tools
        ) 

        if not response.choices[0].message.tool_calls:
            break

        messages.append({
            "role": response.choices[0].message.role,
            "content": response.choices[0].message.content
        })

        # messages.append(
        #     response.choices[0].message.model_dump(exclude_none=True)
        # )

        log(f"\nStep: {step + 1} model requested {len(response.choices[0].message.tool_calls)} tool call(s)\n")

        for tool in response.choices[0].message.tool_calls:

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



