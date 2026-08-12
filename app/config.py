from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API configuration
    host: str = "0.0.0.0"
    port: int = 8000

    mock_llm: bool = True
    
    llm_model_id: str = "Qwen/Qwen1.5-0.5B-Chat"
    stt_model_size: str = "tiny"

    class Config:
        env_file = ".env"

settings = Settings()
