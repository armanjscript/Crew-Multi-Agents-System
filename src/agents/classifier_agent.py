from crewai import Agent, LLM
from config.settings import settings
from src.database.sqlite_client import sqlite_client

class ClassifierAgent:
    def __init__(self, session_id: str = None):
        self.session_id = session_id or "default_session"
        self.llm = LLM(
            base_url=settings.NVIDIA_BASE_URL,
            api_key=settings.NVIDIA_API_KEY,
            model=settings.NVIDIA_MODEL
        )
        
        self.agent = Agent(
            role="Query Classifier with Context",
            goal="Classify user queries considering chat history context",
            backstory="""You are an expert at classifying user queries into specific categories.
            You analyze the user's question and determine whether they are asking about weather,
            want to search for information, or have a general query. You also identify the language
            of the query and maintain the original text. You consider the conversation context
            to make better classification decisions.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def get_agent(self) -> Agent:
        """Return the configured agent"""
        return self.agent
    
    def get_chat_context(self) -> str:
        """Get chat context for classification"""
        recent_messages = sqlite_client.get_recent_context(
            self.session_id, 
            settings.CONTEXT_MESSAGES
        )
        if not recent_messages:
            return "No previous conversation history."
        
        context_parts = []
        for msg in recent_messages:
            role = "User" if msg['type'] == 'human' else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")
        
        return "Recent conversation:\n" + "\n".join(context_parts)