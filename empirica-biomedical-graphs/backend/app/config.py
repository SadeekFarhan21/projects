from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # Database
    database_url: str = "sqlite:///./synapse_mapper.db"
    
    # LLM API Keys (Anthropic only)
    anthropic_api_key: str = ""
    
    # LAVA API Configuration
    lava_secret_key: str = ""
    lava_connection_secret: str = ""
    lava_product_secret: str = ""
    enable_lava: bool = False
    
    # Processing Configuration
    max_upload_size_mb: int = 100
    max_concurrent_processing: int = 4
    enable_llm_extraction: bool = False
    # Use a fine-grained biomedical NER by default
    scispacy_model: str = "en_ner_bionlp13cg_md"
    
    # CORS
    cors_origins: List[str] = [
        "http://localhost:5173", 
        "http://localhost:3000",
        "http://EXAMPLE_HOST:5173",
        "http://EXAMPLE_HOST:3000",
        "https://EXAMPLE_HOST:5173",
        "https://EXAMPLE_HOST:3000",
        "*"  # Allow all origins for development
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

