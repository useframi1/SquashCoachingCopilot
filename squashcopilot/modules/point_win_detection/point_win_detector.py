"""
Point Win Detection for squash rallies.

Assigns point winners based on rally state, hit detection, and court boundaries.
Uses DataFrame-based pipeline architecture.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from squashcopilot.common.utils import load_config
from squashcopilot.common.types.geometry import Point2D
from squashcopilot.common.models import CourtCalibrationOutput


class PointWinDetector:
    """Detects point winners for each rally.
    
    Primary method: Check who serves the next rally (winner serves next)
    Let detection: Same player serves from same side = let/replay
    Fallback method: Ball physics analysis for last rally
    """

    def __init__(self, config: dict = None, calibration: CourtCalibrationOutput = None):
        """Initialize the point win detector.

        Args:
            config: Configuration dictionary. If None, loads from config file
            calibration: Court calibration output with homographies
        """
        if config is None:
            config = load_config(config_name="point_win_detection")

        point_config = config.get("point_win_detection", {})
        
        # Court boundaries
        boundaries = point_config.get("court_boundaries", {})
        self.front_wall_min_y = boundaries.get("front_wall_min_y", 0.0)
        self.front_wall_max_y = boundaries.get("front_wall_max_y", 9.75)
        self.left_wall_min_x = boundaries.get("left_wall_min_x", 0.0)
        self.right_wall_max_x = boundaries.get("right_wall_max_x", 6.4)
        self.tin_height_y = boundaries.get("tin_height_y", 0.48)
        self.out_line_height_y = boundaries.get("out_line_height_y", 4.57)
        self.center_x = boundaries.get("center_x", 3.2)
        
        # Logic settings
        logic = point_config.get("logic", {})
        self.use_next_rally_server = logic.get("use_next_rally_server", True)
        self.detect_lets = logic.get("detect_lets", True)
        self.num_racket_hits_to_check = logic.get("num_racket_hits_to_check", 3)
        self.no_wall_hit_default = logic.get("no_wall_hit_default", "other_player")
        
        self.calibration = calibration

    def load_data_with_rally_state(
        self,
        main_csv_path: Path,
        ground_truth_csv_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """Load CSV data and ensure rally info exists.
        
        Expects CSV with is_rally_frame and rally_id columns.
        
        Args:
            main_csv_path: Path to main annotations CSV
            ground_truth_csv_path: Path to ground truth rally state CSV (optional, for backwards compatibility)
            
        Returns:
            DataFrame with necessary columns including rally info
        """
        print("\n" + "=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        # Load main CSV
        print(f"Loading main CSV: {main_csv_path}")
        df = pd.read_csv(main_csv_path)
        print(f"Loaded {len(df)} frames")
        
        # Set frame as index if not already
        if "frame" in df.columns and df.index.name != "frame":
            df = df.set_index("frame")
        
        # Check for required rally segmentation columns
        if "is_rally_frame" not in df.columns or "rally_id" not in df.columns:
            raise ValueError(
                "CSV must contain 'is_rally_frame' and 'rally_id' columns. "
                "Please use the latest annotated CSV format."
            )
        
        print("✓ Found rally segmentation (is_rally_frame + rally_id)")
        
        # Filter to only rally frames (CSV may already be filtered, but just in case)
        original_len = len(df)
        df = df[df["is_rally_frame"] == True].copy()
        print(f"Filtered to rally frames: {len(df)} frames (from {original_len})")
        
        # Select only necessary columns
        required_cols = [
            "is_rally_frame",
            "rally_id",
            "is_racket_hit",
            "racket_hit_player_id",
            "is_wall_hit",
            "wall_hit_x_meter",
            "wall_hit_y_meter",
            "player_1_x_pixel",
            "player_1_y_pixel",
            "player_1_x_meter",
            "player_1_y_meter",
            "player_2_x_pixel",
            "player_2_y_pixel",
            "player_2_x_meter",
            "player_2_y_meter",
        ]
        
        # Check which columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Keep only required columns
        df = df[required_cols].copy()
        print(f"Kept {len(required_cols)} columns")
        
        # Print statistics
        self._print_data_statistics(df)
        
        return df

    def _print_data_statistics(self, df: pd.DataFrame):
        """Print statistics about loaded data."""
        print("\nData Statistics:")
        print(f"  Total rally frames: {len(df)}")
        
        if "rally_id" in df.columns:
            num_rallies = df["rally_id"].nunique()
            print(f"  Number of rallies: {num_rallies}")
        
        if "is_racket_hit" in df.columns:
            racket_hits = df["is_racket_hit"].sum()
            print(f"  Racket hits: {int(racket_hits)}")
        
        if "is_wall_hit" in df.columns:
            wall_hits = df["is_wall_hit"].sum()
            print(f"  Wall hits: {int(wall_hits)}")

    def extract_rallies(self, df: pd.DataFrame) -> List[Dict]:
        """Extract rally segments from DataFrame based on rally_id.
        
        Args:
            df: DataFrame with rally_id column (already filtered to is_rally_frame=True)
            
        Returns:
            List of rally dictionaries with start_frame, end_frame, rally_id, num_frames
        """
        rallies = []
        
        if "rally_id" not in df.columns:
            print("Warning: No rally_id column found")
            # Treat entire video as one rally
            rallies.append({
                "rally_id": 0,
                "start_frame": int(df.index.min()),
                "end_frame": int(df.index.max()),
                "num_frames": len(df)
            })
            print(f"\nExtracted 1 rally (entire video)")
            return rallies
        
        # Group by rally_id
        for rally_id in sorted(df["rally_id"].unique()):
            # Get all frames for this rally
            rally_df = df[df["rally_id"] == rally_id]
            
            start_frame = int(rally_df.index.min())
            end_frame = int(rally_df.index.max())
            num_frames = len(rally_df)
            
            rallies.append({
                "rally_id": int(rally_id),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "num_frames": num_frames
            })
        
        print(f"\nExtracted {len(rallies)} rallies")
        return rallies

    def get_rally_server(self, rally_df: pd.DataFrame) -> Optional[int]:
        """Determine who serves this rally (first racket hit).
        
        Args:
            rally_df: DataFrame slice for this rally
            
        Returns:
            Player ID (1 or 2) who serves, or None if unknown
        """
        racket_hits = rally_df[rally_df["is_racket_hit"] == True]
        
        if len(racket_hits) == 0:
            return None
        
        # First racket hit = server
        first_hit_frame = racket_hits.index[0]
        server = int(rally_df.loc[first_hit_frame, "racket_hit_player_id"])
        
        return server

    def get_serve_side(self, rally_df: pd.DataFrame) -> Optional[str]:
        """Determine which side of the court the serve came from.
        
        Args:
            rally_df: DataFrame slice for this rally
            
        Returns:
            "left" or "right" based on server's X position, or None if unknown
        """
        racket_hits = rally_df[rally_df["is_racket_hit"] == True]
        
        if len(racket_hits) == 0:
            return None
        
        # Get first racket hit frame
        first_hit_frame = racket_hits.index[0]
        server_id = int(rally_df.loc[first_hit_frame, "racket_hit_player_id"])
        
        # Get server's position at serve
        if server_id == 1:
            server_x = rally_df.loc[first_hit_frame, "player_1_x_meter"]
        else:
            server_x = rally_df.loc[first_hit_frame, "player_2_x_meter"]
        
        if pd.isna(server_x):
            return None
        
        # Determine side based on court center
        return "left" if server_x < self.center_x else "right"

    def is_let_rally(
        self,
        current_rally_df: pd.DataFrame,
        next_rally_df: pd.DataFrame
    ) -> bool:
        """Check if current rally was a let/replay.
        
        A let is detected when the same player serves from the same side
        in consecutive rallies.
        
        Args:
            current_rally_df: DataFrame slice for current rally
            next_rally_df: DataFrame slice for next rally
            
        Returns:
            True if current rally was a let, False otherwise
        """
        if not self.detect_lets:
            return False
        
        # Get server and side for both rallies
        current_server = self.get_rally_server(current_rally_df)
        current_side = self.get_serve_side(current_rally_df)
        
        next_server = self.get_rally_server(next_rally_df)
        next_side = self.get_serve_side(next_rally_df)
        
        # If any information is missing, can't determine let
        if None in [current_server, current_side, next_server, next_side]:
            return False
        
        # Let = same player, same side
        return current_server == next_server and current_side == next_side

    def is_wall_hit_in_bounds(self, wall_hit_x: float, wall_hit_y: float) -> bool:
        """Check if a wall hit is within valid court boundaries.
        
        Args:
            wall_hit_x: Wall hit X coordinate in meters
            wall_hit_y: Wall hit Y coordinate in meters
            
        Returns:
            True if in-bounds, False if out-of-bounds
        """
        # Check if coordinates are valid (not NaN)
        if np.isnan(wall_hit_x) or np.isnan(wall_hit_y):
            return False
        
        # Check X boundaries (side walls)
        if wall_hit_x < self.left_wall_min_x or wall_hit_x > self.right_wall_max_x:
            return False
        
        # Check Y boundaries (front/back walls)
        if wall_hit_y < self.front_wall_min_y or wall_hit_y > self.front_wall_max_y:
            return False
        
        # Check tin (too low on front wall)
        if wall_hit_y < self.tin_height_y:
            return False
        
        # Check out line (too high on front wall)
        if wall_hit_y > self.out_line_height_y:
            return False
        
        return True

    def assign_point_winner_by_next_server(
        self,
        rally_df: pd.DataFrame,
        next_rally_df: pd.DataFrame,
        rally_id: int
    ) -> Tuple[int, str, int]:
        """Assign point winner based on who serves the next rally.
        
        In squash, the winner of a rally serves the next rally.
        If the same player serves from the same side, it was a let/replay.
        
        Args:
            rally_df: DataFrame slice for current rally
            next_rally_df: DataFrame slice for next rally
            rally_id: Rally identifier
            
        Returns:
            Tuple of (winner_player_id, reason, deciding_racket_hit_frame)
            winner_player_id: 0 (let), 1 (player 1), 2 (player 2), or -1 (unknown)
        """
        # Check if this was a let/replay
        if self.is_let_rally(rally_df, next_rally_df):
            # Get first frame of rally as deciding frame for lets
            racket_hits = rally_df[rally_df["is_racket_hit"] == True]
            deciding_frame = racket_hits.index[0] if len(racket_hits) > 0 else rally_df.index[0]
            
            current_server = self.get_rally_server(rally_df)
            current_side = self.get_serve_side(rally_df)
            
            return (
                0,  # 0 indicates a let (no winner)
                f"Let/Replay - Player {current_server} serves again from {current_side} side",
                deciding_frame
            )
        
        # Not a let, get server of next rally = winner of this rally
        next_server = self.get_rally_server(next_rally_df)
        
        if next_server is not None:
            # Find last racket hit of current rally for deciding frame
            racket_hits = rally_df[rally_df["is_racket_hit"] == True]
            deciding_frame = racket_hits.index[-1] if len(racket_hits) > 0 else rally_df.index[0]
            
            return (
                next_server,
                f"Player {next_server} serves next rally (rally {rally_id + 1})",
                deciding_frame
            )
        
        # Fallback if we can't determine next server
        return (-1, "Cannot determine next rally server", rally_df.index[0])

    def assign_point_winner_by_ball_physics(
        self,
        rally_df: pd.DataFrame,
        rally_id: int
    ) -> Tuple[int, str, int]:
        """Assign point winner using ball physics (fallback method).
        
        Checks the last N racket hits to find the point-ending shot.
        
        Args:
            rally_df: DataFrame slice for this rally
            rally_id: Rally identifier
            
        Returns:
            Tuple of (winner_player_id, reason, deciding_racket_hit_frame)
        """
        # Find all racket hits in this rally
        racket_hits_df = rally_df[rally_df["is_racket_hit"] == True]
        
        if len(racket_hits_df) == 0:
            return (-1, "No racket hits detected", rally_df.index[0])
        
        # Get the last N racket hits to check
        num_hits_to_check = min(self.num_racket_hits_to_check, len(racket_hits_df))
        recent_racket_hits = racket_hits_df.index[-num_hits_to_check:]
        
        # Check each racket hit from most recent backwards
        for racket_hit_frame in reversed(recent_racket_hits):
            racket_hit_player = int(rally_df.loc[racket_hit_frame, "racket_hit_player_id"])
            
            # Find wall hits after this racket hit
            frames_after_racket = rally_df.loc[racket_hit_frame:]
            wall_hits_after = frames_after_racket[frames_after_racket["is_wall_hit"] == True]
            
            # Case 1: No wall hit after this racket hit
            if len(wall_hits_after) == 0:
                winner = 2 if racket_hit_player == 1 else 1
                return (
                    winner,
                    f"No wall hit after racket hit by Player {racket_hit_player} (frame {racket_hit_frame})",
                    racket_hit_frame
                )
            
            # Case 2: Wall hit exists, check if it's in bounds
            first_wall_hit_frame = wall_hits_after.index[0]
            wall_hit_x = rally_df.loc[first_wall_hit_frame, "wall_hit_x_meter"]
            wall_hit_y = rally_df.loc[first_wall_hit_frame, "wall_hit_y_meter"]
            
            in_bounds = self.is_wall_hit_in_bounds(wall_hit_x, wall_hit_y)
            
            # If out of bounds, this is the deciding hit
            if not in_bounds:
                winner = 2 if racket_hit_player == 1 else 1
                return (
                    winner,
                    f"Wall hit out-of-bounds at ({wall_hit_x:.2f}, {wall_hit_y:.2f}) after hit by Player {racket_hit_player} (frame {racket_hit_frame})",
                    racket_hit_frame
                )
        
        # If all recent hits are in-bounds, assume last hit won the point
        last_racket_hit_frame = racket_hits_df.index[-1]
        last_racket_hit_player = int(rally_df.loc[last_racket_hit_frame, "racket_hit_player_id"])
        
        # Get wall hit info for reporting
        frames_after_racket = rally_df.loc[last_racket_hit_frame:]
        wall_hits_after = frames_after_racket[frames_after_racket["is_wall_hit"] == True]
        
        if len(wall_hits_after) > 0:
            first_wall_hit_frame = wall_hits_after.index[0]
            wall_hit_x = rally_df.loc[first_wall_hit_frame, "wall_hit_x_meter"]
            wall_hit_y = rally_df.loc[first_wall_hit_frame, "wall_hit_y_meter"]
            reason = f"Wall hit in-bounds at ({wall_hit_x:.2f}, {wall_hit_y:.2f}) by Player {last_racket_hit_player} (frame {last_racket_hit_frame})"
        else:
            reason = f"Rally ended with in-play sequence by Player {last_racket_hit_player} (frame {last_racket_hit_frame})"
        
        return (last_racket_hit_player, reason, last_racket_hit_frame)

    def detect_point_winners(
        self,
        df: pd.DataFrame,
        rallies: List[Dict]
    ) -> Dict[int, Dict]:
        """Detect point winners for all rallies.
        
        Args:
            df: DataFrame with hit detection and rally state
            rallies: List of rally dictionaries
            
        Returns:
            Dictionary mapping rally_id to winner info
            winner: 0 = let/replay, 1 = player 1, 2 = player 2, -1 = unknown
        """
        print("\n" + "=" * 60)
        print("DETECTING POINT WINNERS")
        print("=" * 60)
        
        if self.use_next_rally_server:
            print("Primary method: Using next rally server")
            if self.detect_lets:
                print("Let detection: Enabled (same server, same side)")
            print(f"Fallback method: Checking last {self.num_racket_hits_to_check} racket hits\n")
        else:
            print(f"Method: Checking last {self.num_racket_hits_to_check} racket hits\n")
        
        results = {}
        
        for i, rally in enumerate(rallies):
            rally_id = rally["rally_id"]
            start_frame = rally["start_frame"]
            end_frame = rally["end_frame"]
            
            # Get rally slice
            rally_df = df.loc[start_frame:end_frame]
            
            # Determine winner using appropriate method
            if self.use_next_rally_server and i < len(rallies) - 1:
                # Use next rally's server
                next_rally = rallies[i + 1]
                next_rally_df = df.loc[next_rally["start_frame"]:next_rally["end_frame"]]
                winner, reason, deciding_frame = self.assign_point_winner_by_next_server(
                    rally_df, next_rally_df, rally_id
                )
                method = "next_server"
            else:
                # Use ball physics (for last rally or if disabled)
                winner, reason, deciding_frame = self.assign_point_winner_by_ball_physics(
                    rally_df, rally_id
                )
                method = "ball_physics"
            
            results[rally_id] = {
                "winner": winner,
                "reason": reason,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "num_frames": len(rally_df),
                "deciding_frame": deciding_frame,
                "method": method
            }
            
            # Format winner string for display
            if winner == 0:
                winner_str = "Let/Replay"
            elif winner == -1:
                winner_str = "Unknown"
            else:
                winner_str = f"Player {winner}"
            
            print(f"Rally {rally_id}: {winner_str} [{method}]")
            print(f"  {reason}")
        
        return results

    def run_detection(
        self,
        main_csv_path: Path,
        ground_truth_csv_path: Optional[Path] = None
    ) -> Tuple[pd.DataFrame, List[Dict], Dict[int, Dict]]:
        """Run complete point win detection pipeline.
        
        Args:
            main_csv_path: Path to main annotations CSV
            ground_truth_csv_path: Path to ground truth rally state CSV (optional)
            
        Returns:
            Tuple of (df, rallies, point_winners)
        """
        # Load data
        df = self.load_data_with_rally_state(main_csv_path, ground_truth_csv_path)
        
        # Extract rallies
        rallies = self.extract_rallies(df)
        
        # Detect point winners
        point_winners = self.detect_point_winners(df, rallies)
        
        return df, rallies, point_winners