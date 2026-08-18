from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model configuration (Mocks enabled by default for rapid testing)
    mock_llm: bool = True
    mock_vlm: bool = True
    mock_tts: bool = True
    
    # Model IDs
    llm_model_id: str = "Qwen/Qwen1.5-0.5B-Chat"
    vlm_model_id: str = "llava-hf/llava-1.5-7b-hf"
    stt_model_size: str = "tiny"
    
    class Config:
        env_file = ".env"

settings = Settings()
