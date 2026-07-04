from crewai import Task
from src.models.schemas import (
    ClassifierOutput, WeatherOutput, 
    SearchOutput, GeneralOutput
)
from src.database.sqlite_client import sqlite_client
from config.settings import settings

def create_classifier_task(agent, user_query: str, session_id: str) -> Task:
    """Create task for classifier agent with context"""
    # Get recent conversation context as formatted string
    context_str = sqlite_client.get_recent_context(session_id, settings.CONTEXT_MESSAGES)
    
    return Task(
        description=f"""Analyze the following user query and classify it, considering the conversation context:

        Current User Query: {user_query}
        
        {context_str}
        
        Classify the query into one of these categories:
        - "weather": if the user is asking about weather conditions
        - "search": if the user wants to search for information
        - "general": for greetings, general questions, or when the query doesn't fit the above categories
        
        Consider the conversation context to better understand the query's intent.
        Also identify the language of the query and keep the original query intact.
        
        Provide the output in the specified structured format.""",
        agent=agent,
        expected_output="A JSON object with 'step', 'language', and 'user_original_query' fields",
        output_pydantic=ClassifierOutput
    )

def create_weather_task(agent, user_query: str, language: str = "English") -> Task:
    """Create task for weather agent with language context"""
    return Task(
        description=f"""Extract the city name from the following user query and get weather information.

        User Query: {user_query}
        Query Language: {language}
        
        Instructions:
        1. Identify the specific city mentioned in the query and extract it.
        2. Use the weather tool to get current weather information for the city.
        3. Extract the key weather information (temperature, conditions, etc.).
        4. Format the weather data in a clear, user-friendly way.
        
        Important: You MUST use the weather_api_tool to get actual weather data.
        Do not guess or fabricate weather information.
        
        Provide the output in the specified structured format with both the city name and weather data.""",
        agent=agent,
        expected_output="A JSON object with 'city' and 'weather_data' fields",
        output_pydantic=WeatherOutput
    )

def create_search_task(agent, user_query: str, language: str = "English") -> Task:
    """Create task for search agent with language context"""
    return Task(
        description=f"""Transform the following user query into optimized search keywords and get search results.

        User Query: {user_query}
        Query Language: {language}
        
        Instructions:
        1. Extract at most 4 key keywords from the query that would give the best search results.
        2. Use the google_search_tool to find relevant information.
        3. Extract the most relevant information from the search results.
        4. Format the search results in a clear, organized way.
        
        Important: You MUST use the google_search_tool to get actual search results.
        Do not guess or fabricate information.
        
        Provide the output in the specified structured format with both the transformed query and search results.""",
        agent=agent,
        expected_output="A JSON object with 'transformed_query' and 'search_results' fields",
        output_pydantic=SearchOutput
    )

def create_general_task(agent, user_query: str, session_id: str, language: str = "English", 
                       context_info: str = "") -> Task:
    """Create task for general agent with context and information"""
    # Get recent conversation context
    context_str = sqlite_client.get_recent_context(session_id, settings.CONTEXT_MESSAGES)
    
    return Task(
        description=f"""Handle the following general query with conversation context and provided information.

        Current User Query: {user_query}
        Query Language: {language}
        
        {context_str}
        
        {context_info}
        
        Instructions:
        1. If it's a greeting, respond with an appropriate greeting in the same language.
        2. If the query is about weather or search, use the provided information above to answer.
        3. If you don't have enough information, honestly say "I don't have enough information to answer your questions or query you asked".
        4. Respond in the same language as the user's query whenever possible.
        5. Make your response natural, friendly, and conversational.
        6. Format the information clearly with proper structure.
        
        Make sure your response is contextual and references previous conversation when relevant.
        
        Provide the output in the specified structured format.""",
        agent=agent,
        expected_output="A JSON object with 'response' field",
        output_pydantic=GeneralOutput
    )