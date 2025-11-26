"""LLM conversation model for storing chat history and context."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship

from backend.models.database import Base

if TYPE_CHECKING:
    from backend.models.video import Video


class LLMConversation(Base):
    """Model for storing LLM conversation history and context."""

    __tablename__ = "llm_conversations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=True)
    player_id = Column(Integer, nullable=True)  # 1 or 2
    messages = Column(JSON, nullable=False, default=list)  # Array of conversation messages
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    video = relationship("Video", back_populates="llm_conversations")

    def __repr__(self) -> str:
        return f"<LLMConversation(id={self.id}, video_id={self.video_id}, messages={len(self.messages)})>"
