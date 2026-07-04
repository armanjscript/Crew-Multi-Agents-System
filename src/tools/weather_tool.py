import requests
from typing import Dict, Any
from crewai.tools import BaseTool
from config.settings import settings

class WeatherTool(BaseTool):
    name: str = "weather_api_tool"
    description: str = "Get current weather information for a city"
    
    def _run(self, city: str) -> Dict[str, Any]:
        """Get weather data for a given city"""
        try:
            params = {
                "key": settings.WEATHER_API_KEY,
                "q": city,
                "aqi": "no"
            }
            response = requests.get(settings.WEATHER_API_URL, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Failed to fetch weather data: {str(e)}"}
    
    async def _arun(self, city: str) -> Dict[str, Any]:
        """Async version of weather API call"""
        return self._run(city)