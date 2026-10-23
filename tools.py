from datetime import datetime
import requests

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

name_to_fn = {
    "get_current_time": get_current_time,
    "get_weather_temperature": get_weather_temperature,
    "get_user_city": get_user_city
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
    }
]