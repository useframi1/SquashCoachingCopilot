"""High-level coaching analysis combining all metrics."""

from typing import List, Dict, Optional
import json
import os
from data.data_models import FrameData
from .movement_analyzer import MovementAnalyzer
from .rally_analyzer import RallyAnalyzer


class CoachingAnalyzer:
    """
    High-level analyzer for coaching insights.

    Responsibilities:
    - Combine movement and rally analyses
    - Generate coaching recommendations
    - Provide performance comparisons
    - Export analysis results for coaching app
    """

    def __init__(self, fps: float = 30.0):
        """
        Initialize coaching analyzer.

        Args:
            fps: Video frames per second
        """
        self.fps = fps
        self.movement_analyzer = MovementAnalyzer(fps=fps)
        self.rally_analyzer = RallyAnalyzer(fps=fps)

    def analyze_match(self, frames: List[FrameData]) -> Dict:
        """
        Perform comprehensive match analysis.

        Args:
            frames: List of frame data from entire match

        Returns:
            Dictionary with complete match analysis
        """
        # Movement analysis
        movement_metrics = self.movement_analyzer.analyze_both_players(frames)

        # Rally analysis
        rally_stats = self.rally_analyzer.get_rally_statistics(frames)
        all_rallies = self.rally_analyzer.analyze_all_rallies(frames)

        # Match metadata
        match_info = {
            "total_frames": len(frames),
            "duration": len(frames) / self.fps,
            "fps": self.fps,
        }

        # Combine all analyses
        analysis = {
            "match_info": match_info,
            "movement_analysis": movement_metrics,
            "rally_statistics": rally_stats,
            "rallies": all_rallies,
            "coaching_insights": self._generate_insights(
                movement_metrics, rally_stats, all_rallies
            ),
        }

        return analysis

    def analyze_player_performance(
        self, frames: List[FrameData], player_id: int
    ) -> Dict:
        """
        Analyze individual player performance.

        Args:
            frames: List of frame data
            player_id: Player ID (1 or 2)

        Returns:
            Dictionary with player-specific analysis
        """
        # Movement metrics
        movement = self.movement_analyzer.analyze_player_movement(frames, player_id)

        # Speed and acceleration profiles
        speed_profile = self.movement_analyzer.get_speed_profile(frames, player_id)
        accel_profile = self.movement_analyzer.get_acceleration_profile(frames, player_id)

        # Sprint detection
        sprints = self.movement_analyzer.detect_sprints(frames, player_id)

        # Positioning analysis
        positioning = self.movement_analyzer.analyze_positioning(frames, player_id)

        return {
            "player_id": player_id,
            "movement_metrics": movement,
            "speed_profile_summary": {
                "max": max(speed_profile) if speed_profile else 0.0,
                "average": sum(speed_profile) / len(speed_profile) if speed_profile else 0.0,
            },
            "acceleration_profile_summary": {
                "max": max(accel_profile) if accel_profile else 0.0,
                "min": min(accel_profile) if accel_profile else 0.0,
            },
            "sprints": {
                "count": len(sprints),
                "details": sprints,
            },
            "positioning": positioning,
        }

    def compare_players(self, frames: List[FrameData]) -> Dict:
        """
        Compare performance metrics between both players.

        Args:
            frames: List of frame data

        Returns:
            Dictionary with comparative analysis
        """
        p1_analysis = self.analyze_player_performance(frames, player_id=1)
        p2_analysis = self.analyze_player_performance(frames, player_id=2)

        comparison = {
            "player1": p1_analysis,
            "player2": p2_analysis,
            "comparison_summary": {
                "more_active_player": 1 if p1_analysis["movement_metrics"]["total_distance"] > p2_analysis["movement_metrics"]["total_distance"] else 2,
                "faster_player": 1 if p1_analysis["movement_metrics"]["max_speed"] > p2_analysis["movement_metrics"]["max_speed"] else 2,
                "better_court_coverage": 1 if p1_analysis["positioning"]["court_coverage"] > p2_analysis["positioning"]["court_coverage"] else 2,
            },
        }

        return comparison

    def export_analysis(
        self, frames: List[FrameData], output_path: str, format: str = "json"
    ):
        """
        Export analysis results to file.

        Args:
            frames: List of frame data
            output_path: Path to save analysis
            format: Export format ('json' or 'summary')
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        analysis = self.analyze_match(frames)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(analysis, f, indent=2, default=str)
        elif format == "summary":
            self._export_text_summary(analysis, output_path)

        print(f"Analysis exported to: {output_path}")

    def _generate_insights(
        self, movement_metrics: Dict, rally_stats: Dict, rallies: List[Dict]
    ) -> List[str]:
        """
        Generate coaching insights based on analysis.

        Args:
            movement_metrics: Movement analysis results
            rally_stats: Rally statistics
            rallies: Individual rally analyses

        Returns:
            List of coaching insight strings
        """
        insights = []

        # Movement insights
        p1_coverage = movement_metrics["player1"]["court_coverage"]
        p2_coverage = movement_metrics["player2"]["court_coverage"]

        if p1_coverage < 30:
            insights.append("Player 1: Low court coverage. Consider improving movement to cover more court area.")
        if p2_coverage < 30:
            insights.append("Player 2: Low court coverage. Consider improving movement to cover more court area.")

        # Speed insights
        p1_speed = movement_metrics["player1"]["average_speed"]
        p2_speed = movement_metrics["player2"]["average_speed"]

        if p1_speed > p2_speed * 1.2:
            insights.append("Player 1 is significantly more active, potentially dominating rallies.")
        elif p2_speed > p1_speed * 1.2:
            insights.append("Player 2 is significantly more active, potentially dominating rallies.")

        # Rally insights
        if rally_stats["average_duration"] > 15:
            insights.append("Long rally duration indicates good fitness and defensive play.")
        elif rally_stats["average_duration"] < 5:
            insights.append("Short rallies suggest aggressive play or quick errors.")

        # Intensity comparison
        if rally_stats.get("player1_avg_intensity", 0) > rally_stats.get("player2_avg_intensity", 0) * 1.3:
            insights.append("Player 1 shows higher movement intensity during rallies.")
        elif rally_stats.get("player2_avg_intensity", 0) > rally_stats.get("player1_avg_intensity", 0) * 1.3:
            insights.append("Player 2 shows higher movement intensity during rallies.")

        if not insights:
            insights.append("Performance metrics are balanced. Continue current training approach.")

        return insights

    def _export_text_summary(self, analysis: Dict, output_path: str):
        """Export analysis as human-readable text summary."""
        with open(output_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("SQUASH MATCH ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            # Match info
            f.write("MATCH INFORMATION\n")
            f.write("-" * 60 + "\n")
            f.write(f"Duration: {analysis['match_info']['duration']:.1f} seconds\n")
            f.write(f"Total Frames: {analysis['match_info']['total_frames']}\n\n")

            # Rally stats
            f.write("RALLY STATISTICS\n")
            f.write("-" * 60 + "\n")
            rally_stats = analysis["rally_statistics"]
            f.write(f"Total Rallies: {rally_stats['total_rallies']}\n")
            f.write(f"Average Duration: {rally_stats['average_duration']:.2f}s\n")
            f.write(f"Average Shots/Rally: {rally_stats['average_shots_per_rally']:.1f}\n\n")

            # Movement analysis
            f.write("MOVEMENT ANALYSIS\n")
            f.write("-" * 60 + "\n")
            for player_num in [1, 2]:
                player_key = f"player{player_num}"
                metrics = analysis["movement_analysis"][player_key]
                f.write(f"\nPlayer {player_num}:\n")
                f.write(f"  Total Distance: {metrics['total_distance']:.2f}m\n")
                f.write(f"  Average Speed: {metrics['average_speed']:.2f}m/s\n")
                f.write(f"  Court Coverage: {metrics['court_coverage']:.1f}%\n")
                f.write(f"  Direction Changes: {metrics['direction_changes']}\n")

            # Coaching insights
            f.write("\n" + "=" * 60 + "\n")
            f.write("COACHING INSIGHTS\n")
            f.write("=" * 60 + "\n")
            for i, insight in enumerate(analysis["coaching_insights"], 1):
                f.write(f"{i}. {insight}\n")
