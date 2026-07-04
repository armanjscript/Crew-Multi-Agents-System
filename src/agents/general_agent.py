from crewai import Agent, LLM
from config.settings import settings
from typing import List, Dict, Any

class GeneralAgent:
    def __init__(self):
        self.llm = LLM(
            base_url=settings.NVIDIA_BASE_URL,
            api_key=settings.NVIDIA_API_KEY,
            model=settings.NVIDIA_MODEL
        )
        
        self.agent = Agent(
            role="General Assistant with Context",
            goal="Handle general queries with context from chat history",
            backstory="""You are a helpful assistant that handles general queries with awareness 
            of the conversation context. You use the last 3 messages from chat history to provide 
            more relevant and contextual responses. You:
            1. Greet users appropriately
            2. Answer questions based on your knowledge and conversation context
            3. Honestly admit when you don't have enough information
            4. Maintain conversation flow by referencing previous exchanges
            You're friendly, helpful, and transparent about your capabilities.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def get_agent(self) -> Agent:
        """Return the configured agent"""
        return self.agent
    
    def format_context(self, chat_history: List[Dict[str, Any]]) -> str:
        """Format chat history for context"""
        if not chat_history:
            return "No previous conversation history."
        
        formatted_messages = []
        for msg in chat_history:
            role = "User" if msg['type'] == 'human' else "Assistant"
            formatted_messages.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted_messages)