"""High-level coaching analysis combining all metrics."""

from typing import List, Dict, Optional
import json
import os
from data.data_models import RallyData, convert_numpy_types


class CoachingAnalyzer:
    """
    High-level analyzer for coaching insights.

    Responsibilities:
    - Combine movement and rally analyses
    - Generate coaching recommendations
    - Provide performance comparisons
    - Export analysis results for coaching app
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
        Perform comprehensive match analysis.

        Args:
            rallies: List of rally data from entire match

        Returns:
            Dictionary with complete match analysis
        """
        # Calculate average rally duration
        avg_rally_duration = sum(
            [(r.end_frame - r.start_frame) / self.fps for r in rallies]
        ) / len(rallies)

        # Combine all analyses
        analysis = {
            "fps": self.fps,
            "total_rallies": len(rallies),
            "avg_rally_duration": convert_numpy_types(avg_rally_duration),
            "rallies": [r.to_dict() for r in rallies],
        }

        return analysis
