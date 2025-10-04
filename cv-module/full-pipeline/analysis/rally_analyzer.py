"""Rally-level analysis for game insights."""

from typing import List, Dict, Optional, Tuple
import numpy as np
from data.data_models import FrameData


class RallyAnalyzer:
    """
    Analyzes rally-level metrics and patterns.

    Responsibilities:
    - Segment rallies from frame data
    - Calculate rally duration and intensity
    - Analyze rally outcomes and patterns
    - Provide tactical insights from rally data
    """

    def __init__(self, fps: float = 30.0):
        """
        Initialize rally analyzer.

        Args:
            fps: Video frames per second
        """
        self.fps = fps

    def segment_rallies(self, frames: List[FrameData]) -> List[Dict]:
        """
        Segment video into individual rallies based on rally state.

        Args:
            frames: List of frame data

        Returns:
            List of rally segments with start/end frames and state info
        """
        rallies = []
        current_rally = None

        for i, frame in enumerate(frames):
            rally_state = frame.rally_state

            # Detect rally start
            if rally_state == "Rally" and (current_rally is None or current_rally["state"] != "Rally"):
                if current_rally is not None:
                    # End previous rally
                    current_rally["end_frame"] = i - 1
                    rallies.append(current_rally)

                # Start new rally
                current_rally = {
                    "start_frame": i,
                    "end_frame": None,
                    "state": "Rally",
                }

            # Detect rally end
            elif rally_state != "Rally" and current_rally is not None and current_rally["state"] == "Rally":
                current_rally["end_frame"] = i - 1
                rallies.append(current_rally)
                current_rally = None

        # Close last rally if still open
        if current_rally is not None:
            current_rally["end_frame"] = len(frames) - 1
            rallies.append(current_rally)

        return rallies

    def analyze_rally(
        self, frames: List[FrameData], rally_segment: Dict
    ) -> Dict:
        """
        Analyze a single rally segment.

        Args:
            frames: List of frame data
            rally_segment: Rally segment info (start_frame, end_frame)

        Returns:
            Dictionary with rally analysis metrics
        """
        start = rally_segment["start_frame"]
        end = rally_segment["end_frame"]
        rally_frames = frames[start:end + 1]

        # Calculate basic metrics
        duration = len(rally_frames) / self.fps

        # Calculate player distances traveled
        p1_distance = self._calculate_player_distance(rally_frames, player_id=1)
        p2_distance = self._calculate_player_distance(rally_frames, player_id=2)

        # Calculate rally intensity (average speed)
        p1_intensity = p1_distance / duration if duration > 0 else 0.0
        p2_intensity = p2_distance / duration if duration > 0 else 0.0

        # Calculate shot count estimate (based on ball position changes)
        shot_count = self._estimate_shot_count(rally_frames)

        return {
            "start_frame": start,
            "end_frame": end,
            "duration": duration,
            "player1_distance": p1_distance,
            "player2_distance": p2_distance,
            "player1_intensity": p1_intensity,
            "player2_intensity": p2_intensity,
            "shot_count": shot_count,
        }

    def analyze_all_rallies(self, frames: List[FrameData]) -> List[Dict]:
        """
        Analyze all rallies in the video.

        Args:
            frames: List of frame data

        Returns:
            List of rally analysis results
        """
        rally_segments = self.segment_rallies(frames)
        return [self.analyze_rally(frames, segment) for segment in rally_segments]

    def get_rally_statistics(self, frames: List[FrameData]) -> Dict:
        """
        Get overall rally statistics for the match.

        Args:
            frames: List of frame data

        Returns:
            Dictionary with aggregate rally statistics
        """
        rally_analyses = self.analyze_all_rallies(frames)

        if not rally_analyses:
            return self._empty_rally_stats()

        durations = [r["duration"] for r in rally_analyses]
        p1_distances = [r["player1_distance"] for r in rally_analyses]
        p2_distances = [r["player2_distance"] for r in rally_analyses]
        p1_intensities = [r["player1_intensity"] for r in rally_analyses]
        p2_intensities = [r["player2_intensity"] for r in rally_analyses]
        shot_counts = [r["shot_count"] for r in rally_analyses]

        return {
            "total_rallies": len(rally_analyses),
            "average_duration": np.mean(durations),
            "max_duration": max(durations),
            "min_duration": min(durations),
            "total_play_time": sum(durations),
            "player1_avg_distance": np.mean(p1_distances),
            "player2_avg_distance": np.mean(p2_distances),
            "player1_avg_intensity": np.mean(p1_intensities),
            "player2_avg_intensity": np.mean(p2_intensities),
            "average_shots_per_rally": np.mean(shot_counts),
        }

    def _calculate_player_distance(
        self, frames: List[FrameData], player_id: int
    ) -> float:
        """Calculate total distance traveled by player in rally."""
        trajectory = []

        for frame in frames:
            player = frame.get_player(player_id)
            if player and player.is_valid() and player.real_position:
                trajectory.append(player.real_position)

        if len(trajectory) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i - 1][0]
            dy = trajectory[i][1] - trajectory[i - 1][1]
            total_distance += np.sqrt(dx ** 2 + dy ** 2)

        return total_distance

    def _estimate_shot_count(self, frames: List[FrameData]) -> int:
        """
        Estimate number of shots in rally based on ball trajectory changes.

        This is a simple heuristic - can be improved with more sophisticated
        shot detection algorithms.
        """
        # Extract ball positions
        ball_positions = []
        for frame in frames:
            if frame.ball.is_valid():
                ball_positions.append(frame.ball.position)

        if len(ball_positions) < 10:
            return 0

        # Detect direction changes in ball trajectory as proxy for shots
        shot_count = 0
        for i in range(2, len(ball_positions)):
            # Calculate velocity vectors
            v1 = (
                ball_positions[i - 1][0] - ball_positions[i - 2][0],
                ball_positions[i - 1][1] - ball_positions[i - 2][1],
            )
            v2 = (
                ball_positions[i][0] - ball_positions[i - 1][0],
                ball_positions[i][1] - ball_positions[i - 1][1],
            )

            # Calculate angle between vectors
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_angle))

                # Significant direction change suggests a shot
                if angle_deg > 90:
                    shot_count += 1

        return shot_count

    def _empty_rally_stats(self) -> Dict:
        """Return empty rally statistics structure."""
        return {
            "total_rallies": 0,
            "average_duration": 0.0,
            "max_duration": 0.0,
            "min_duration": 0.0,
            "total_play_time": 0.0,
            "player1_avg_distance": 0.0,
            "player2_avg_distance": 0.0,
            "player1_avg_intensity": 0.0,
            "player2_avg_intensity": 0.0,
            "average_shots_per_rally": 0.0,
        }
