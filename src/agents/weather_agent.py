from crewai import Agent, LLM
from config.settings import settings
from src.tools.weather_tool import WeatherTool

class WeatherAgent:
    def __init__(self):
        self.llm = LLM(
            base_url=settings.NVIDIA_BASE_URL,
            api_key=settings.NVIDIA_API_KEY,
            model=settings.NVIDIA_MODEL,
        )
        
        self.weather_tool = WeatherTool()
        
        self.agent = Agent(
            role="Weather Specialist",
            goal="Extract city names and provide weather information",
            backstory="""You are a weather specialist who extracts city names from user queries
            and fetches current weather information. You're precise at identifying location names
            and providing accurate weather data.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.weather_tool],
            llm=self.llm
        )
    
    def get_agent(self) -> Agent:
        """Return the configured agent"""
        return self.agent