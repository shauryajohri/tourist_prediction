import os
import requests
from dotenv import load_dotenv

load_dotenv()

def get_weather_on_date(location, date):
    api_key = os.environ.get("WEATHER_API_KEY")
    if not api_key:
        return {"error": "WEATHER_API_KEY not set. Add it to your .env file."}
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    url = f"{base_url}/{location}/{date}?unitGroup=metric&key={api_key}&include=days"

    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "days" in data and data["days"]:
            day_weather = data["days"][0]
            return {
                "date": day_weather["datetime"],
                "temp": day_weather["temp"],
                "conditions": day_weather["conditions"],
                "description": day_weather.get("description", "N/A"),
                "humidity": day_weather.get("humidity", "N/A"),
                "windspeed": day_weather.get("windspeed", "N/A")
            }
        else:
            return {"error": "No weather data available for this date."}
    else:
        return {"error": f"Failed to fetch data. Status code: {response.status_code}"}

