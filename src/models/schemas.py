from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any

class ClassifierOutput(BaseModel):
    """Structured output for the classifier agent"""
    step: Literal["weather", "search", "general"] = Field(
        description="Classified step based on user query"
    )
    language: str = Field(
        description="Original language of the user query"
    )
    user_original_query: str = Field(
        description="The original user query"
    )

class WeatherOutput(BaseModel):
    """Structured output for weather agent"""
    city: str = Field(
        description="City name extracted from the user query"
    )
    weather_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Weather data from API"
    )

class SearchOutput(BaseModel):
    """Structured output for search agent"""
    transformed_query: str = Field(
        description="Transformed query with up to 4 keywords"
    )
    search_results: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Search results from the search tool"
    )

class GeneralOutput(BaseModel):
    """Structured output for general agent"""
    response: str = Field(
        description="The response to the user query"
    )

class ConversationPair(BaseModel):
    """Model for conversation pairs"""
    id: Optional[int] = None
    session_id: str
    human_message: str
    ai_message: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

class ChatHistory(BaseModel):
    """Model for chat history"""
    session_id: str
    pairs: List[ConversationPair]