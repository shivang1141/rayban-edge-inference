from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API configuration
    host: str = "0.0.0.0"
    port: int = 8000

    stt_model_size: str = "tiny"

    class Config:
        env_file = ".env"

settings = Settings()
