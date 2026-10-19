from openai import OpenAI 
from dotenv import load_dotenv
import os, json
from tools import llm_tools, get_current_time, name_to_fn

load_dotenv()

base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")
model = os.getenv("MODEL")


client = OpenAI(
    base_url=base_url,
    api_key=api_key
)



messages = []

messages.append({
    "role": "system",
    "content": "You are a helpful assistant who is a bit sarcastic"
})

while True:

    prompt = input("\nuser: ")
    if prompt.lower() in ["exit", "bye"]:
        break
    
    messages.append({
        "role": "user",
        "content": prompt
    })

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=llm_tools
    ) 

    messages.append({
        "role": response.choices[0].message.role,
        "content": response.choices[0].message.content
    })

    if response.choices[0].message.tool_calls:

        for tool in response.choices[0].message.tool_calls:

            args = json.loads(tool.function.arguments) or {}
            data = name_to_fn[tool.function.name](**args)

            messages.append({
                "role": "tool",
                "tool_id": tool.id,
                "content": str(data)
            })

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=llm_tools
        ) 

    messages.append({
        "role": response.choices[0].message.role,
        "content": response.choices[0].message.content
    })
    
    print("qwen: " + response.choices[0].message.content + "\n")



