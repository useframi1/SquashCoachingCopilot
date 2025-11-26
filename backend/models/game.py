"""Game model for tracking individual games within a match."""

from sqlalchemy import Column, String, Float, Integer, ForeignKey, Index
from sqlalchemy.orm import relationship

from backend.models.database import Base


class Game(Base):
    """Individual game within a match."""

    __tablename__ = "games"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False, index=True)
    game_number = Column(Integer, nullable=False)  # 1, 2, 3, 4, 5

    # Game result
    winner = Column(Integer, nullable=True)  # 1 or 2
    player_1_score = Column(Integer, nullable=False, default=0)
    player_2_score = Column(Integer, nullable=False, default=0)

    # Rally range for this game
    start_rally_id = Column(Integer, nullable=False)
    end_rally_id = Column(Integer, nullable=False)

    # Timestamps
    start_time = Column(Float, nullable=True)
    end_time = Column(Float, nullable=True)

    # Relationships
    video = relationship("Video", back_populates="games")

    # Composite indexes for efficient queries
    __table_args__ = (Index("idx_video_game", "video_id", "game_number"),)

    def __repr__(self) -> str:
        return f"<Game(video_id={self.video_id}, game={self.game_number}, score={self.player_1_score}-{self.player_2_score})>"
