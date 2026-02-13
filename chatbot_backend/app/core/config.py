from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Base URLs
    base_url: str = "http://localhost:8000"
    express_base_url: str = "http://localhost:3000"

    # Database
    mongo_atlas_uri: str
    db_name: str = "CVProject"

    # Google OAuth
    google_redirect_uri: Optional[str] = None

    # AI Service
    ai_service_url: Optional[str] = None
    ai_service_api_key: str = "ai-service-internal-key"

    @property
    def google_redirect_url(self) -> str:
        return self.google_redirect_uri or f"{self.base_url}/auth/google/callback"

    @property
    def ai_service_base_url(self) -> str:
        return self.ai_service_url or f"{self.express_base_url}/business/api"

    # LLM
    default_model: str = "qwen2.5:3b"
    gemini_api_key: Optional[str] = None
    ollama_api_base: str

    # Application
    debug: bool = False
    log_level: str = "INFO"
    port: int = 8000
    host: str = "0.0.0.0"

    # CORS
    cors_origins: List[str]

    # File Upload
    max_file_size: int = 10485760  # 10MB
    upload_dir: str = "uploads"

    # OCR
    tesseract_cmd: Optional[str] = None
    poppler_path: Optional[str] = None

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}


settings = Settings()
