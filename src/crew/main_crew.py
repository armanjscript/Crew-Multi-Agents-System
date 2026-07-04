from crewai import Crew, Process
from typing import Dict, Any, Optional, List
import json
from src.agents.classifier_agent import ClassifierAgent
from src.agents.weather_agent import WeatherAgent
from src.agents.search_agent import SearchAgent
from src.agents.general_agent import GeneralAgent
from src.tasks.task_definitions import (
    create_classifier_task,
    create_weather_task,
    create_search_task,
    create_general_task
)
from src.models.schemas import ClassifierOutput, WeatherOutput, SearchOutput
from src.database.sqlite_client import sqlite_client
from config.settings import settings
import uuid

class MainCrew:
    def __init__(self, session_id: Optional[str] = None):
        # Generate or use provided session ID
        self.session_id = session_id or str(uuid.uuid4())
        
        # Initialize all agents with session ID
        self.classifier_agent = ClassifierAgent(self.session_id).get_agent()
        self.weather_agent = WeatherAgent().get_agent()
        self.search_agent = SearchAgent().get_agent()
        self.general_agent = GeneralAgent().get_agent()
        
        # Initialize general agent for context formatting
        self.general_agent_instance = GeneralAgent()
        
        print(f"📝 Session ID: {self.session_id}")
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process a user query through the multi-agent system"""
        
        # Step 1: Classify the query with context
        classifier_task = create_classifier_task(
            self.classifier_agent, 
            user_query,
            self.session_id
        )
        
        classifier_crew = Crew(
            agents=[self.classifier_agent],
            tasks=[classifier_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            classification_result = classifier_crew.kickoff()
            
            # Extract the raw string output from CrewOutput
            raw_output = classification_result.raw if hasattr(classification_result, 'raw') else str(classification_result)
            
            print(f"📝 Raw classifier output: {raw_output}")
            
            # Parse classification result from the raw string
            classifier_data = ClassifierOutput.model_validate_json(raw_output)
            
            print(f"✅ Classifier Result: step={classifier_data.step}, language={classifier_data.language}")
            print(f"📝 Original Query: {classifier_data.user_original_query}")
            print("=" * 50)
            
        except Exception as e:
            print(f"❌ Classification failed: {str(e)}")
            classifier_data = ClassifierOutput(
                step="general",
                language="unknown",
                user_original_query=user_query
            )
        
        # Extract values from classifier
        step = classifier_data.step
        original_query = classifier_data.user_original_query
        language = classifier_data.language
        
        print(f"🔄 Routing to: {step.upper()} agent")
        print(f"🌐 Language: {language}")
        print(f"💬 Query: {original_query}")
        print("=" * 50)
        
        # Step 2: Route to appropriate agent based on classification
        result = None
        context_info = ""
        metadata = {"step": step, "language": language}
        
        if step == "weather":
            print("🌤️ Calling Weather Agent...")
            result = self._handle_weather(original_query, language)
            # Extract weather information for context
            context_info = self._format_weather_context(result)
            metadata["weather_data"] = context_info
        elif step == "search":
            print("🔍 Calling Search Agent...")
            result = self._handle_search(original_query, language)
            # Extract search information for context
            context_info = self._format_search_context(result)
            metadata["search_data"] = context_info
        else:  # general
            print("💬 Calling General Agent...")
            result = self._handle_general(original_query, language, "")
        
        # If weather or search, pass the context to general agent for final response
        if step in ["weather", "search"] and result:
            print("📝 Passing information to General Agent for final response...")
            final_response = self._handle_general(original_query, language, context_info)
            final_result = final_response
        else:
            final_result = result
        
        # Parse and save AI response
        try:
            if isinstance(final_result, str):
                try:
                    result_data = json.loads(final_result)
                    if isinstance(result_data, dict) and 'response' in result_data:
                        ai_response = result_data['response']
                    else:
                        ai_response = final_result
                except json.JSONDecodeError:
                    ai_response = final_result
            else:
                ai_response = str(final_result)
        except:
            ai_response = str(final_result)
        
        # Save conversation pair to database
        sqlite_client.add_conversation_pair(
            session_id=self.session_id,
            human_message=user_query,
            ai_message=ai_response,
            metadata=metadata
        )
        
        return {
            "session_id": self.session_id,
            "classification": classifier_data.model_dump(),
            "result": final_result
        }
    
    def _handle_weather(self, user_query: str, language: str = "English") -> Dict[str, Any]:
        """Handle weather queries with language context"""
        weather_task = create_weather_task(
            self.weather_agent, 
            user_query,
            language
        )
        
        weather_crew = Crew(
            agents=[self.weather_agent],
            tasks=[weather_task],
            process=Process.sequential,
            verbose=True
        )
        
        result = weather_crew.kickoff()
        
        # Parse the result to extract weather data
        try:
            raw_output = result.raw if hasattr(result, 'raw') else str(result)
            weather_data = WeatherOutput.model_validate_json(raw_output)
            return weather_data
        except:
            return result
    
    def _handle_search(self, user_query: str, language: str = "English") -> Dict[str, Any]:
        """Handle search queries with language context"""
        search_task = create_search_task(
            self.search_agent, 
            user_query,
            language
        )
        
        search_crew = Crew(
            agents=[self.search_agent],
            tasks=[search_task],
            process=Process.sequential,
            verbose=True
        )
        
        result = search_crew.kickoff()
        
        # Parse the result to extract search data
        try:
            raw_output = result.raw if hasattr(result, 'raw') else str(result)
            search_data = SearchOutput.model_validate_json(raw_output)
            return search_data
        except:
            return result
    
    def _handle_general(self, user_query: str, language: str = "English", context_info: str = "") -> Dict[str, Any]:
        """Handle general queries with context and language"""
        general_task = create_general_task(
            self.general_agent, 
            user_query,
            self.session_id,
            language,
            context_info
        )
        
        general_crew = Crew(
            agents=[self.general_agent],
            tasks=[general_task],
            process=Process.sequential,
            verbose=True
        )
        
        return general_crew.kickoff()
    
    def _format_weather_context(self, weather_data: Any) -> str:
        """Format weather data for context"""
        try:
            if hasattr(weather_data, 'weather_data') and weather_data.weather_data:
                data = weather_data.weather_data
                city = weather_data.city
                
                # Extract relevant weather information
                if isinstance(data, dict):
                    temp = data.get('current', {}).get('temp_c', 'N/A')
                    condition = data.get('current', {}).get('condition', {}).get('text', 'N/A')
                    humidity = data.get('current', {}).get('humidity', 'N/A')
                    wind = data.get('current', {}).get('wind_kph', 'N/A')
                    
                    return f"""Weather Information for {city}:
                    - Temperature: {temp}°C
                    - Condition: {condition}
                    - Humidity: {humidity}%
                    - Wind Speed: {wind} km/h"""
            
            return "Weather information retrieved successfully."
        except:
            return "Weather information retrieved successfully."
    
    def _format_search_context(self, search_data: Any) -> str:
        """Format search results for context"""
        try:
            if hasattr(search_data, 'search_results') and search_data.search_results:
                results = search_data.search_results
                query = search_data.transformed_query
                
                formatted_results = f"Search Results for '{query}':\n"
                
                for i, result in enumerate(results[:3], 1):  # Show top 3 results
                    title = result.get('title', 'No title')
                    snippet = result.get('snippet', 'No description available')
                    formatted_results += f"\n{i}. {title}\n   {snippet}\n"
                
                return formatted_results
            
            return "Search results retrieved successfully."
        except:
            return "Search results retrieved successfully."
    
    def get_chat_history(self, limit: Optional[int] = None) -> list:
        """Get chat history as formatted messages"""
        return sqlite_client.get_chat_history(self.session_id, limit)
    
    def get_conversation_pairs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation pairs"""
        return sqlite_client.get_recent_pairs(self.session_id, limit if limit else 999)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for current session"""
        return sqlite_client.get_session_stats(self.session_id)
    
    def clear_session(self) -> None:
        """Clear all messages for current session"""
        sqlite_client.clear_session(self.session_id)
    
    def delete_session(self) -> None:
        """Delete current session"""
        sqlite_client.delete_session(self.session_id)
    
    def get_context(self) -> str:
        """Get formatted context for debugging"""
        return sqlite_client.get_recent_context(self.session_id, settings.CONTEXT_MESSAGES)
    
    def export_session(self) -> Dict[str, Any]:
        """Export current session data"""
        return sqlite_client.export_session(self.session_id)