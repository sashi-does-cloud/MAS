from datetime import datetime
from utils.util import log, error, warn
from rag import query, load_chunks

import json
import requests

def get_current_time():
    return {
        "current_time": datetime.now().strftime("%I:%M:%S %p"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timezone": "Asia/Kolkata"
    }
    
def get_weather_temperature(city):
    url = f"https://wttr.in/{city}?format=3"
    response = requests.get(url)
    # will raise HTTPError if response status code is 4xx or 5xx
    response.raise_for_status()
    return response.text

def get_user_city():
    # url = "https://ipapi.co/json/"
    # data = requests.get(url, timeout=10).json()
    # return data['city'] + ", " + data['region_code'] + ", " + data['country_name']
    return "tirupati"

def run_tool(tool):
    args = json.loads(tool.function.arguments) or {}
    try:
        func_name = tool.function.name
        func_args = json.dumps(args)

        log(f"🔧 {func_name}({func_args})")
        data = name_to_fn[func_name](**args)

        log(f"   ↳ {data}")

    except Exception as e:
        data = {"error": str(e)}
        warn(f"Error while running {func_name} : {e}")
    
    return data

def search_knowledge_base(search_query, chunks=None, top_k=3):
    log(f" search_knowledge_base(search_query={search_query!r}, top_k={top_k})")
    if chunks is None:
        log("   ↳ No chunks provided, loading from vector store...")
        chunks = load_chunks()
    log(f"   ↳ Searching over {len(chunks)} chunks")
    result = query(search_query, chunks, top_k)
    log(f"   ↳ Done. Returned top {top_k} results")

    # Blocks is for providing context to LLM
    blocks = []

    for chunk in result:
        chunk_id = chunk["chunk_id"]
        source = chunk["source"]
        text = chunk["text"]
        score = chunk["score"]

        log(f"  {score:.2f} - {source} #{chunk_id}")

        header = f"{source} #{chunk_id}"
        block = f"{header}\n{text}"

        blocks.append(block)

    return blocks


name_to_fn = {
    "get_current_time": get_current_time,
    "get_weather_temperature": get_weather_temperature,
    "get_user_city": get_user_city,
    "search_knowledge_base": search_knowledge_base
}

llm_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current time in the format HH:MM using 12-hour format with am/pm",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_temperature",
            "description": "Retrieves current weather conditions and weather forecasts for a specified location using the wttr.in weather service. Accepts a city name, airport code, or other supported location identifier. Returns weather information such as temperature, feels-like temperature, weather conditions, humidity, wind speed, and forecast data when available.\n\nUse this tool whenever the user asks about current weather, temperature, weather conditions, or upcoming forecasts for a particular location. Extract the location from the user's query and pass it to the tool. If no location is provided, ask the user to specify one unless the user's location is already available to the agent.\n\nInterpret the returned data accurately. Do not invent missing weather information or claim the forecast is real-time if the response does not support that.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_city",
            "description": "Returns the city where the user lives. Use this when the user says 'my city', 'my Home', 'where I live' or 'my location' without naming the city.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Searches the internal company knowledge base using semantic similarity "
                "and retrieves the most relevant information. The knowledge base contains "
                "the following company documents:\n\n"
                "- **Leave Policy**: Casual leave (12 days/year), earned leave (18 days/year, "
                "carry-forward up to 30), sick leave (10 days), maternity (26 weeks) and "
                "paternity leave (10 days), and leave application via PeopleHub portal.\n"
                "- **Travel Expense Policy**: Reimbursement rules, flight class guidelines "
                "(economy vs business class), daily allowances (domestic and international), "
                "and receipt requirements.\n"
                "- **Medical Policy**: Health insurance coverage (5 lakh family floater), "
                "cashless treatment, long medical leave, maternity hospitalization cover, "
                "and annual health checkup.\n"
                "- **HR Policies**: Working hours (10 AM–4 PM core), work-from-home rules, "
                "onboarding process, performance review cycle (April & October), and notice "
                "period (60 days standard).\n"
                "- **Hiring Process**: Interview stages (5 rounds), offer letter timeline, "
                "referral bonus (25,000 rupees), and new joiner onboarding.\n"
                "- **Tech Docs**: Code review guidelines (2 approvals, <400 lines), "
                "deployment windows (Tue/Thu 11 AM–4 PM), branch naming conventions, "
                "and on-call rotation.\n"
                "- **RCA / Incident Reports**: Post-mortem reports for production incidents "
                "(e.g. payment service outage INC-4021) including timeline, root cause, fix, "
                "and prevention steps.\n\n"
                "Use this tool whenever the user asks about any company policy, HR rule, "
                "engineering guideline, or incident report. Extract a concise, focused search "
                "query from the user's question and pass it to the tool. If the topic is "
                "unclear, ask the user to clarify before searching."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": (
                            "A short, focused search query capturing the key topic from the "
                            "user's message (e.g. 'maternity leave duration', 'business class "
                            "flight approval', 'health insurance coverage limit', "
                            "'deployment window production', 'referral bonus amount')."
                        )
                    },
                    "chunks": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": (
                            "Pre-loaded list of document chunks with embeddings to search over. "
                            "Each chunk is an object with 'text', 'source', 'chunk_id', and "
                            "'embedding' fields. If not provided, chunks are loaded automatically "
                            "from the vector store."
                        )
                    },
                    "top_k": {
                        "type": "integer",
                        "description": (
                            "Number of top matching chunks to return. Defaults to 3. "
                            "Increase for broader coverage, decrease for a tighter result."
                        )
                    }
                },
                "required": ["search_query", "chunks"]
            }
        }
    }
]