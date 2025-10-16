"""Simplified coaching analysis for rally count and player positioning."""

from typing import List, Dict
from data.data_models import RallyData


class CoachingAnalyzer:
    """
    Simplified analyzer for basic coaching insights.

    Responsibilities:
    - Count rallies in a match
    - Calculate average positioning for players
    """

    def __init__(self, fps: float):
        """
        Initialize coaching analyzer.

        Args:
            fps: Video frames per second
        """
        self.fps = fps

    def analyze_match(self, rallies: List[RallyData]) -> Dict:
        """
        Perform simplified match analysis.

        Args:
            rallies: List of rally data from entire match

        Returns:
            Dictionary with rally count and average positioning
        """
        # Count rallies
        rally_count = len(rallies)

        # Calculate average positioning for both players
        player1_positions = []
        player2_positions = []

        for rally in rallies:
            for frame in rally.rally_frames:
                if frame.player1.real_position:
                    player1_positions.append(frame.player1.real_position)

                if frame.player2.real_position:
                    player2_positions.append(frame.player2.real_position)

        # Calculate averages
        avg_player1_position = None
        if player1_positions:
            avg_x = sum(pos[0] for pos in player1_positions) / len(player1_positions)
            avg_y = sum(pos[1] for pos in player1_positions) / len(player1_positions)
            avg_player1_position = {"x": avg_x, "y": avg_y}

        avg_player2_position = None
        if player2_positions:
            avg_x = sum(pos[0] for pos in player2_positions) / len(player2_positions)
            avg_y = sum(pos[1] for pos in player2_positions) / len(player2_positions)
            avg_player2_position = {"x": avg_x, "y": avg_y}

        return {
            "rally_count": rally_count,
            "average_positioning": {
                "player1": avg_player1_position,
                "player2": avg_player2_position,
            },
        }
