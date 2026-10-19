from datetime import datetime
import requests

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_weather_temperature(city, value="2"):
    url = f"https://wttr.in/{city}?format={value}"
    response = requests.get(url)
    return response.text

name_to_fn = {
    "get_current_time": get_current_time,
    "get_weather_temperature": get_weather_temperature
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
                    },
                    "value": {
                        "type": "number",
                        "description": "Report format types: 1. Concise (3 lines), 2. Standard (6 lines), 3. Detailed (10 lines)"
                    }
                },
                "required": ["city"]
            }
        }
    }
]