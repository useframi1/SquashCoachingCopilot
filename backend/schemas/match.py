"""Pydantic schemas for match and game data."""

from typing import List, Optional
from pydantic import BaseModel, Field


class GameSchema(BaseModel):
    """Schema for a single game."""

    id: int
    video_id: str
    game_number: int
    winner: Optional[int] = Field(None, description="Player who won the game (1 or 2)")
    player_1_score: int
    player_2_score: int
    start_rally_id: int
    end_rally_id: int
    start_time: Optional[float]
    end_time: Optional[float]

    class Config:
        from_attributes = True


class MatchSchema(BaseModel):
    """Schema for match-level summary."""

    id: int
    video_id: str
    winner: Optional[int] = Field(None, description="Player who won the match (1 or 2)")
    player_1_games_won: int
    player_2_games_won: int
    total_rallies: int
    total_games: int
    scoring_system: str = Field(default="11-point", description="Scoring system (11-point or 9-point)")
    best_of: int = Field(default=5, description="Best of how many games (3 or 5)")

    class Config:
        from_attributes = True


class MatchSummaryResponse(BaseModel):
    """Complete match summary with games."""

    video_id: str
    match: MatchSchema
    games: List[GameSchema]

    class Config:
        from_attributes = True
