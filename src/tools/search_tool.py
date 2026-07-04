import requests
from typing import Dict, Any, List
from crewai.tools import BaseTool
from config.settings import settings

class SerperSearchTool(BaseTool):
    name: str = "google_search_tool"
    description: str = "Search the web using Google Serper API"
    
    def _run(self, query: str) -> List[Dict[str, Any]]:
        """Perform a web search using Serper API"""
        try:
            headers = {
                "X-API-KEY": settings.SERPER_API_KEY,
                "Content-Type": "application/json"
            }
            data = {
                "q": query,
                "num": 5  # Number of results to return
            }
            response = requests.post(
                "https://google.serper.dev/search",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            return result.get("organic", [])
        except Exception as e:
            return [{"error": f"Failed to perform search: {str(e)}"}]
    
    async def _arun(self, query: str) -> List[Dict[str, Any]]:
        """Async version of search"""
        return self._run(query)