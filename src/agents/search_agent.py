from crewai import Agent, LLM
from config.settings import settings
from src.tools.search_tool import SerperSearchTool

class SearchAgent:
    def __init__(self):
        self.llm = LLM(
            base_url=settings.NVIDIA_BASE_URL,
            api_key=settings.NVIDIA_API_KEY,
            model=settings.NVIDIA_MODEL
        )
        
        self.search_tool = SerperSearchTool()
        
        self.agent = Agent(
            role="Search Specialist",
            goal="Transform queries into search-friendly keywords",
            backstory="""You are an expert at transforming user queries into optimized search keywords.
            You extract the most relevant 3-4 keywords from the user's original query to get the best
            search results. You're efficient and precise at identifying key information.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.search_tool],
            llm=self.llm
        )
    
    def get_agent(self) -> Agent:
        """Return the configured agent"""
        return self.agent