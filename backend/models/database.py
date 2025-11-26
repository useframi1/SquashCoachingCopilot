"""Database connection and session management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from backend.config import settings

# Create engine
# PostgreSQL connection pooling configuration
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,  # Verify connections before using them
    pool_size=10,  # Number of connections to maintain
    max_overflow=20,  # Additional connections if pool is full
    echo=settings.debug,
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()


def get_db():
    """Dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables."""
    # Import all models to ensure they're registered with Base.metadata
    from backend.models.video import Video  # noqa: F401
    from backend.models.job import Job  # noqa: F401
    from backend.models.frame_data import FrameData  # noqa: F401
    from backend.models.game import Game  # noqa: F401
    from backend.models.match import Match  # noqa: F401
    from backend.models.llm_conversation import LLMConversation  # noqa: F401

    Base.metadata.create_all(bind=engine)
