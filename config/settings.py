import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # NVIDIA LLM Configuration
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    NVIDIA_MODEL = "nvidia_nim/z-ai/glm-5.2"
    
    # Weather API Configuration
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
    WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"
    
    # Google Serper API Configuration
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    
    # SQLite Configuration
    DATABASE_PATH = os.getenv("DATABASE_PATH", "data/chat_history.db")
    
    # Chat History Settings
    MAX_HISTORY_MESSAGES = 100  # Maximum messages to store per session
    CONTEXT_MESSAGES = 3  # Number of recent messages to use for context
    
    @classmethod
    def validate_settings(cls):
        """Validate that all required API keys are present"""
        required_keys = [
            ("NVIDIA_API_KEY", cls.NVIDIA_API_KEY),
            ("WEATHER_API_KEY", cls.WEATHER_API_KEY),
            ("SERPER_API_KEY", cls.SERPER_API_KEY),
        ]
        missing_keys = [key for key, value in required_keys if not value]
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(cls.DATABASE_PATH), exist_ok=True)

settings = Settings()