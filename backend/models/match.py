"""Match model for tracking match-level results."""

from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship

from backend.models.database import Base


class Match(Base):
    """Match-level summary (one per video)."""

    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False, unique=True, index=True)

    # Match result
    winner = Column(Integer, nullable=True)  # 1 or 2, None if incomplete
    player_1_games_won = Column(Integer, nullable=False, default=0)
    player_2_games_won = Column(Integer, nullable=False, default=0)

    # Total stats
    total_rallies = Column(Integer, nullable=False)
    total_games = Column(Integer, nullable=False)

    # Match format (11-point or 9-point, best of 3 or 5)
    scoring_system = Column(String(20), default="11-point")  # "11-point" or "9-point"
    best_of = Column(Integer, default=5)  # 3 or 5

    # Relationships
    video = relationship("Video", back_populates="match")

    def __repr__(self) -> str:
        return f"<Match(video_id={self.video_id}, games={self.player_1_games_won}-{self.player_2_games_won})>"
