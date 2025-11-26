"""
Application configuration using Pydantic Settings.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "SquashCoachingCopilot API"
    debug: bool = False

    # Database (PostgreSQL)
    database_url: str = (
        "postgresql://squash_user:squash_password@localhost:5432/squash_copilot"
    )

    # Storage paths
    upload_dir: Path = Path("backend/storage/uploads")
    temp_dir: Path = Path("backend/storage/temp")

    # Video processing
    max_video_size_mb: int = 500
    allowed_video_extensions: set = {".mp4", ".mov", ".avi", ".mkv"}
    max_fps: int = 30  # Maximum FPS - videos will be converted if they exceed this
    enable_fps_conversion: bool = (
        True  # Set to False to disable automatic FPS conversion
    )

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Pipeline settings
    pipeline_timeout_seconds: int = 3600  # 1 hour max for processing

    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-5-nano", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=1000, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def setup_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        # No need to create database directory for PostgreSQL (managed by Docker)


# Global settings instance
settings = Settings()
