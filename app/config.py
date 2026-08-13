from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API configuration
    host: str = "0.0.0.0"
    port: int = 8000

    mock_llm: bool = True
    mock_vlm: bool = True
    
    llm_model_id: str = "Qwen/Qwen1.5-0.5B-Chat"
    vlm_model_id: str = "llava-hf/llava-1.5-7b-hf"
    stt_model_size: str = "tiny"

    class Config:
        env_file = ".env"

settings = Settings()
