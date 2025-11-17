"""
Rally State Analyzer

Analyzes ground truth data to understand patterns distinguishing rally from non-rally states.
Provides comprehensive metrics and visualizations for ball trajectory characteristics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from squashcopilot.common.utils import load_config
from squashcopilot.common.models.rally import RallySegment


@dataclass
class VelocityMetrics:
    """Velocity-based metrics for a state."""

    mean_abs_velocity: float
    median_abs_velocity: float
    std_velocity: float
    max_velocity: float
    min_velocity: float
    large_movement_count: int
    large_movement_percentage: float
    stationary_frame_count: int
    stationary_frame_percentage: float
    velocity_percentiles: Dict[int, float]


@dataclass
class PositionMetrics:
    """Position-based metrics for a state."""

    mean_position: float
    std_position: float
    trajectory_range: float
    min_position: float
    max_position: float
    mean_abs_deviation: float
    position_percentiles: Dict[int, float]


@dataclass
class MotionPatternMetrics:
    """Motion pattern metrics for a state."""

    velocity_autocorrelations: Dict[int, float]
    mean_abs_acceleration: float
    std_acceleration: float
    max_acceleration: float
    coefficient_of_variation: float
    high_acceleration_count: int
    high_acceleration_percentage: float


@dataclass
class OscillationMetrics:
    """Oscillation pattern metrics for a state."""

    direction_changes: int
    direction_change_rate: float
    mean_oscillation_amplitude: float
    median_oscillation_amplitude: float
    mean_time_between_changes: float


@dataclass
class StateMetrics:
    """Complete metrics for a single state (rally or non-rally)."""

    state_name: str
    total_frames: int
    velocity: Optional[VelocityMetrics] = None
    position: Optional[PositionMetrics] = None
    motion: Optional[MotionPatternMetrics] = None
    oscillation: Optional[OscillationMetrics] = None


@dataclass
class SegmentAnalysis:
    """Analysis of individual rally/non-rally segments."""

    segment_id: int
    state: str
    start_frame: int
    end_frame: int
    duration: int
    mean_velocity: float
    std_velocity: float
    trajectory_range: float
    large_movement_percentage: float


class RallyStateAnalyzer:
    """
    Analyzes ground truth data to understand patterns in rally vs non-rally states.

    This class loads ground truth annotations and computes comprehensive metrics
    to characterize ball trajectory patterns during rallies and non-rally periods.
    All parameters are configurable via YAML configuration.
    """

    def __init__(self, config_name: str = "rally_state_detection"):
        """
        Initialize analyzer with configuration.

        Args:
            config_name: Name of the configuration file (without .yaml extension)
        """
        # Load full config
        full_config = load_config(config_name=config_name)
        self.config = full_config["analysis"]
        self.annotation_config = full_config["annotation"]

        # Video and paths
        self.video_name = self.config["video_name"]
        project_root = Path(__file__).parent.parent.parent.parent

        # Data directory
        data_base_dir = project_root / self.config["data_dir"]
        self.data_dir = data_base_dir / self.video_name

        # Output directory
        output_base_dir = project_root / self.config["output_dir"]
        self.output_dir = output_base_dir / self.video_name / "analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load analysis parameters
        self.windows = self.config["windows"]
        self.thresholds = self.config["thresholds"]
        self.compute_metrics_flags = self.config["compute_metrics"]
        self.viz_config = self.config["visualization"]
        self.stats_config = self.config["statistics"]

        # Cache for loaded data
        self._annotations_df = None
        self._ground_truth_segments = None
        self._velocity = None
        self._acceleration = None

    # ============================================================================
    # Data Loading Methods
    # ============================================================================

    def load_annotations(self) -> pd.DataFrame:
        """
        Load rally state annotations from CSV file.

        Returns:
            DataFrame with frame, ball_y, and rally state labels
        """
        if self._annotations_df is not None:
            return self._annotations_df

        csv_path = self.data_dir / f"{self.video_name}_annotations.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {csv_path}")

        self._annotations_df = pd.read_csv(csv_path)
        print(f"Loaded {len(self._annotations_df)} annotated frames from {csv_path}")

        return self._annotations_df

    def extract_ground_truth_segments(
        self, df: pd.DataFrame = None
    ) -> Tuple[List[RallySegment], List[RallySegment]]:
        """
        Extract ground truth rally and non-rally segments from annotations.

        Args:
            df: DataFrame with annotations. If None, loads from file.

        Returns:
            Tuple of (rally_segments, non_rally_segments)
        """
        if self._ground_truth_segments is not None:
            return self._ground_truth_segments

        if df is None:
            df = self.load_annotations()

        label_col = self.annotation_config["label_column"]
        rally_segments = []
        non_rally_segments = []

        # Track current state
        current_state = df.iloc[0][label_col]
        segment_start = df.iloc[0]["frame"]
        segment_id = 0

        for i in range(1, len(df)):
            frame = df.iloc[i]["frame"]
            state = df.iloc[i][label_col]

            # Check for state transition
            if state != current_state:
                # End current segment
                segment_end = df.iloc[i - 1]["frame"]
                segment = RallySegment(
                    rally_id=segment_id,
                    start_frame=segment_start,
                    end_frame=segment_end,
                )

                if current_state == "start":
                    rally_segments.append(segment)
                else:
                    non_rally_segments.append(segment)

                # Start new segment
                segment_id += 1
                segment_start = frame
                current_state = state

        # Handle last segment
        segment_end = df.iloc[-1]["frame"]
        segment = RallySegment(
            rally_id=segment_id, start_frame=segment_start, end_frame=segment_end
        )
        if current_state == "start":
            rally_segments.append(segment)
        else:
            non_rally_segments.append(segment)

        self._ground_truth_segments = (rally_segments, non_rally_segments)
        print(
            f"Extracted {len(rally_segments)} rally segments and "
            f"{len(non_rally_segments)} non-rally segments"
        )

        return rally_segments, non_rally_segments

    def compute_velocity(self, df: pd.DataFrame = None) -> np.ndarray:
        """
        Compute frame-to-frame velocity.

        Args:
            df: DataFrame with ball_y positions. If None, loads from file.

        Returns:
            Array of velocities (px/frame)
        """
        if self._velocity is not None:
            return self._velocity

        if df is None:
            df = self.load_annotations()

        positions = df["ball_y"].values
        velocity = np.diff(positions, prepend=positions[0])
        self._velocity = velocity

        return velocity

    def compute_acceleration(self, df: pd.DataFrame = None) -> np.ndarray:
        """
        Compute frame-to-frame acceleration.

        Args:
            df: DataFrame with ball_y positions. If None, loads from file.

        Returns:
            Array of accelerations (px/frame^2)
        """
        if self._acceleration is not None:
            return self._acceleration

        velocity = self.compute_velocity(df)
        acceleration = np.diff(velocity, prepend=velocity[0])
        self._acceleration = acceleration

        return acceleration

    # ============================================================================
    # Metric Computation Methods
    # ============================================================================

    def compute_velocity_metrics(
        self, velocity: np.ndarray, state_mask: np.ndarray
    ) -> VelocityMetrics:
        """
        Compute velocity-based metrics for a given state.

        Args:
            velocity: Array of velocities
            state_mask: Boolean mask for the state

        Returns:
            VelocityMetrics object
        """
        vel_state = velocity[state_mask]
        abs_vel = np.abs(vel_state)

        # Large movement analysis
        large_movements = abs_vel > self.thresholds["large_movement"]
        large_count = np.sum(large_movements)
        large_pct = (large_count / len(vel_state)) * 100 if len(vel_state) > 0 else 0

        # Stationary frame analysis
        stationary = abs_vel < self.thresholds["stationary"]
        stationary_count = np.sum(stationary)
        stationary_pct = (
            (stationary_count / len(vel_state)) * 100 if len(vel_state) > 0 else 0
        )

        # Percentiles
        percentiles = {
            p: np.percentile(abs_vel, p) for p in self.stats_config["percentiles"]
        }

        return VelocityMetrics(
            mean_abs_velocity=float(np.mean(abs_vel)),
            median_abs_velocity=float(np.median(abs_vel)),
            std_velocity=float(np.std(vel_state)),
            max_velocity=float(np.max(abs_vel)),
            min_velocity=float(np.min(abs_vel)),
            large_movement_count=int(large_count),
            large_movement_percentage=float(large_pct),
            stationary_frame_count=int(stationary_count),
            stationary_frame_percentage=float(stationary_pct),
            velocity_percentiles=percentiles,
        )

    def compute_position_metrics(
        self, positions: np.ndarray, state_mask: np.ndarray
    ) -> PositionMetrics:
        """
        Compute position-based metrics for a given state.

        Args:
            positions: Array of ball positions
            state_mask: Boolean mask for the state

        Returns:
            PositionMetrics object
        """
        pos_state = positions[state_mask]
        mean_pos = np.mean(pos_state)

        # Percentiles
        percentiles = {
            p: np.percentile(pos_state, p) for p in self.stats_config["percentiles"]
        }

        return PositionMetrics(
            mean_position=float(mean_pos),
            std_position=float(np.std(pos_state)),
            trajectory_range=float(np.max(pos_state) - np.min(pos_state)),
            min_position=float(np.min(pos_state)),
            max_position=float(np.max(pos_state)),
            mean_abs_deviation=float(np.mean(np.abs(pos_state - mean_pos))),
            position_percentiles=percentiles,
        )

    def compute_motion_pattern_metrics(
        self, velocity: np.ndarray, acceleration: np.ndarray, state_mask: np.ndarray
    ) -> MotionPatternMetrics:
        """
        Compute motion pattern metrics for a given state.

        Args:
            velocity: Array of velocities
            acceleration: Array of accelerations
            state_mask: Boolean mask for the state

        Returns:
            MotionPatternMetrics object
        """
        vel_state = velocity[state_mask]
        acc_state = acceleration[state_mask]
        abs_acc = np.abs(acc_state)

        # Velocity autocorrelation
        autocorrs = {}
        for lag in self.stats_config["autocorrelation_lags"]:
            if len(vel_state) > lag:
                autocorr = np.corrcoef(vel_state[:-lag], vel_state[lag:])[0, 1]
                autocorrs[lag] = float(autocorr) if not np.isnan(autocorr) else 0.0
            else:
                autocorrs[lag] = 0.0

        # Coefficient of variation
        mean_vel = np.mean(np.abs(vel_state))
        std_vel = np.std(vel_state)
        cv = (std_vel / mean_vel) if mean_vel > 0 else 0.0

        # High acceleration analysis
        high_acc = abs_acc > self.thresholds["high_acceleration"]
        high_acc_count = np.sum(high_acc)
        high_acc_pct = (
            (high_acc_count / len(acc_state)) * 100 if len(acc_state) > 0 else 0
        )

        return MotionPatternMetrics(
            velocity_autocorrelations=autocorrs,
            mean_abs_acceleration=float(np.mean(abs_acc)),
            std_acceleration=float(np.std(acc_state)),
            max_acceleration=float(np.max(abs_acc)),
            coefficient_of_variation=float(cv),
            high_acceleration_count=int(high_acc_count),
            high_acceleration_percentage=float(high_acc_pct),
        )

    def compute_oscillation_metrics(
        self, velocity: np.ndarray, state_mask: np.ndarray
    ) -> OscillationMetrics:
        """
        Compute oscillation pattern metrics for a given state.

        Args:
            velocity: Array of velocities
            state_mask: Boolean mask for the state

        Returns:
            OscillationMetrics object
        """
        vel_state = velocity[state_mask]

        # Find direction changes
        signs = np.sign(vel_state)
        sign_changes = np.diff(signs) != 0
        direction_changes = np.sum(sign_changes)
        direction_change_rate = (
            (direction_changes / len(vel_state)) * 100 if len(vel_state) > 0 else 0
        )

        # Oscillation amplitude (distance between peaks and valleys)
        change_indices = np.where(sign_changes)[0]
        amplitudes = []
        if len(change_indices) > 1:
            # Get positions at direction changes
            positions_at_changes = np.cumsum(vel_state)[change_indices]
            amplitudes = np.abs(np.diff(positions_at_changes))

        mean_amplitude = float(np.mean(amplitudes)) if len(amplitudes) > 0 else 0.0
        median_amplitude = float(np.median(amplitudes)) if len(amplitudes) > 0 else 0.0

        # Time between direction changes
        time_between_changes = (
            float(np.mean(np.diff(change_indices))) if len(change_indices) > 1 else 0.0
        )

        return OscillationMetrics(
            direction_changes=int(direction_changes),
            direction_change_rate=float(direction_change_rate),
            mean_oscillation_amplitude=mean_amplitude,
            median_oscillation_amplitude=median_amplitude,
            mean_time_between_changes=time_between_changes,
        )

    def analyze_state(
        self, state_name: str, state_mask: np.ndarray, df: pd.DataFrame = None
    ) -> StateMetrics:
        """
        Compute all metrics for a given state.

        Args:
            state_name: Name of the state ("rally" or "non-rally")
            state_mask: Boolean mask for the state
            df: DataFrame with annotations. If None, loads from file.

        Returns:
            StateMetrics object with all computed metrics
        """
        if df is None:
            df = self.load_annotations()

        positions = df["ball_y"].values
        velocity = self.compute_velocity(df)
        acceleration = self.compute_acceleration(df)

        metrics = StateMetrics(
            state_name=state_name, total_frames=int(np.sum(state_mask))
        )

        # Compute each category of metrics based on config
        if self.compute_metrics_flags["velocity"]:
            metrics.velocity = self.compute_velocity_metrics(velocity, state_mask)

        if self.compute_metrics_flags["position"]:
            metrics.position = self.compute_position_metrics(positions, state_mask)

        if self.compute_metrics_flags["motion_patterns"]:
            metrics.motion = self.compute_motion_pattern_metrics(
                velocity, acceleration, state_mask
            )

        if self.compute_metrics_flags["oscillation"]:
            metrics.oscillation = self.compute_oscillation_metrics(velocity, state_mask)

        return metrics

    def analyze_segments(
        self, df: pd.DataFrame = None
    ) -> Tuple[List[SegmentAnalysis], List[SegmentAnalysis]]:
        """
        Analyze individual rally and non-rally segments.

        Args:
            df: DataFrame with annotations. If None, loads from file.

        Returns:
            Tuple of (rally_segment_analyses, non_rally_segment_analyses)
        """
        if df is None:
            df = self.load_annotations()

        rally_segments, non_rally_segments = self.extract_ground_truth_segments(df)
        velocity = self.compute_velocity(df)
        positions = df["ball_y"].values

        def analyze_segment(segment: RallySegment, state: str) -> SegmentAnalysis:
            # Get indices for this segment
            mask = (df["frame"] >= segment.start_frame) & (
                df["frame"] <= segment.end_frame
            )
            seg_velocity = velocity[mask]
            seg_positions = positions[mask]

            # Compute metrics
            abs_vel = np.abs(seg_velocity)
            large_movements = np.sum(abs_vel > self.thresholds["large_movement"])
            large_pct = (
                (large_movements / len(seg_velocity)) * 100
                if len(seg_velocity) > 0
                else 0
            )

            return SegmentAnalysis(
                segment_id=segment.rally_id,
                state=state,
                start_frame=segment.start_frame,
                end_frame=segment.end_frame,
                duration=segment.duration_frames,
                mean_velocity=float(np.mean(abs_vel)),
                std_velocity=float(np.std(seg_velocity)),
                trajectory_range=float(np.max(seg_positions) - np.min(seg_positions)),
                large_movement_percentage=float(large_pct),
            )

        rally_analyses = [analyze_segment(seg, "rally") for seg in rally_segments]
        non_rally_analyses = [
            analyze_segment(seg, "non-rally") for seg in non_rally_segments
        ]

        return rally_analyses, non_rally_analyses

    # ============================================================================
    # Visualization Methods
    # ============================================================================

    def plot_velocity_distributions(
        self,
        rally_metrics: StateMetrics,
        non_rally_metrics: StateMetrics,
        save: bool = True,
    ):
        """
        Plot velocity distribution comparison between rally and non-rally states.

        Args:
            rally_metrics: Metrics for rally state
            non_rally_metrics: Metrics for non-rally state
            save: Whether to save the plot
        """
        df = self.load_annotations()
        velocity = self.compute_velocity(df)
        label_col = self.annotation_config["label_column"]
        rally_mask = df[label_col] == "start"

        rally_vel = np.abs(velocity[rally_mask])
        non_rally_vel = np.abs(velocity[~rally_mask])

        fig, axes = plt.subplots(1, 2, figsize=self.viz_config["figure_size_medium"])

        # Histogram
        ax1 = axes[0]
        bins = np.linspace(
            0,
            max(rally_vel.max(), non_rally_vel.max()),
            self.viz_config["histogram_bins"],
        )
        ax1.hist(
            rally_vel,
            bins=bins,
            alpha=0.6,
            label="Rally",
            color=self.viz_config["colors"]["rally"],
            edgecolor="black",
        )
        ax1.hist(
            non_rally_vel,
            bins=bins,
            alpha=0.6,
            label="Non-Rally",
            color=self.viz_config["colors"]["non_rally"],
            edgecolor="black",
        )
        ax1.set_xlabel("Absolute Velocity (px/frame)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Frequency", fontsize=12, fontweight="bold")
        ax1.set_title("Velocity Distribution", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2 = axes[1]
        bp = ax2.boxplot(
            [rally_vel, non_rally_vel],
            labels=["Rally", "Non-Rally"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor(self.viz_config["colors"]["rally"])
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(self.viz_config["colors"]["non_rally"])
        bp["boxes"][1].set_alpha(0.6)
        ax2.set_ylabel("Absolute Velocity (px/frame)", fontsize=12, fontweight="bold")
        ax2.set_title("Velocity Statistics", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        # Add statistics
        stats_text = (
            f"Rally:\n"
            f"  Mean: {rally_metrics.velocity.mean_abs_velocity:.2f}\n"
            f"  Median: {rally_metrics.velocity.median_abs_velocity:.2f}\n"
            f"  Std: {rally_metrics.velocity.std_velocity:.2f}\n\n"
            f"Non-Rally:\n"
            f"  Mean: {non_rally_metrics.velocity.mean_abs_velocity:.2f}\n"
            f"  Median: {non_rally_metrics.velocity.median_abs_velocity:.2f}\n"
            f"  Std: {non_rally_metrics.velocity.std_velocity:.2f}"
        )
        ax2.text(
            0.02,
            0.98,
            stats_text,
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "velocity_distributions.png"
            plt.savefig(output_path, dpi=self.viz_config["dpi"], bbox_inches="tight")
            print(f"Velocity distribution plot saved to: {output_path}")
            plt.close(fig)
        else:
            plt.show()

    def plot_position_distributions(
        self,
        rally_metrics: StateMetrics,
        non_rally_metrics: StateMetrics,
        save: bool = True,
    ):
        """
        Plot position distribution comparison between rally and non-rally states.

        Args:
            rally_metrics: Metrics for rally state
            non_rally_metrics: Metrics for non-rally state
            save: Whether to save the plot
        """
        df = self.load_annotations()
        positions = df["ball_y"].values
        label_col = self.annotation_config["label_column"]
        rally_mask = df[label_col] == "start"

        rally_pos = positions[rally_mask]
        non_rally_pos = positions[~rally_mask]

        fig, axes = plt.subplots(1, 2, figsize=self.viz_config["figure_size_medium"])

        # Histogram
        ax1 = axes[0]
        bins = np.linspace(
            min(rally_pos.min(), non_rally_pos.min()),
            max(rally_pos.max(), non_rally_pos.max()),
            self.viz_config["histogram_bins"],
        )
        ax1.hist(
            rally_pos,
            bins=bins,
            alpha=0.6,
            label="Rally",
            color=self.viz_config["colors"]["rally"],
            edgecolor="black",
        )
        ax1.hist(
            non_rally_pos,
            bins=bins,
            alpha=0.6,
            label="Non-Rally",
            color=self.viz_config["colors"]["non_rally"],
            edgecolor="black",
        )
        ax1.set_xlabel("Ball Y Position (px)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Frequency", fontsize=12, fontweight="bold")
        ax1.set_title("Position Distribution", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2 = axes[1]
        bp = ax2.boxplot(
            [rally_pos, non_rally_pos], labels=["Rally", "Non-Rally"], patch_artist=True
        )
        bp["boxes"][0].set_facecolor(self.viz_config["colors"]["rally"])
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(self.viz_config["colors"]["non_rally"])
        bp["boxes"][1].set_alpha(0.6)
        ax2.set_ylabel("Ball Y Position (px)", fontsize=12, fontweight="bold")
        ax2.set_title("Position Statistics", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        # Add statistics
        stats_text = (
            f"Rally:\n"
            f"  Range: {rally_metrics.position.trajectory_range:.1f}\n"
            f"  Mean: {rally_metrics.position.mean_position:.1f}\n"
            f"  Std: {rally_metrics.position.std_position:.1f}\n\n"
            f"Non-Rally:\n"
            f"  Range: {non_rally_metrics.position.trajectory_range:.1f}\n"
            f"  Mean: {non_rally_metrics.position.mean_position:.1f}\n"
            f"  Std: {non_rally_metrics.position.std_position:.1f}"
        )
        ax2.text(
            0.02,
            0.98,
            stats_text,
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "position_distributions.png"
            plt.savefig(output_path, dpi=self.viz_config["dpi"], bbox_inches="tight")
            print(f"Position distribution plot saved to: {output_path}")
            plt.close(fig)
        else:
            plt.show()

    def plot_trajectory_with_states(self, save: bool = True):
        """
        Plot ball trajectory with rally/non-rally states highlighted.

        Args:
            save: Whether to save the plot
        """
        df = self.load_annotations()
        rally_segments, non_rally_segments = self.extract_ground_truth_segments(df)

        fig, ax = plt.subplots(figsize=self.viz_config["figure_size_large"])

        # Plot trajectory
        ax.plot(
            df["frame"],
            df["ball_y"],
            linewidth=1.5,
            color="blue",
            alpha=0.7,
            label="Ball Trajectory",
        )

        # Highlight rally segments
        for segment in rally_segments:
            ax.axvspan(
                segment.start_frame,
                segment.end_frame,
                alpha=0.3,
                color=self.viz_config["colors"]["rally"],
                label="Rally" if segment.rally_id == 0 else "",
            )

        # Highlight non-rally segments
        for segment in non_rally_segments:
            ax.axvspan(
                segment.start_frame,
                segment.end_frame,
                alpha=0.3,
                color=self.viz_config["colors"]["non_rally"],
                label="Non-Rally" if segment.rally_id == 0 else "",
            )

        ax.set_xlabel("Frame Number", fontsize=12, fontweight="bold")
        ax.set_ylabel("Ball Y Position (px)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Ball Trajectory with Rally States - {self.video_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "trajectory_with_states.png"
            plt.savefig(output_path, dpi=self.viz_config["dpi"], bbox_inches="tight")
            print(f"Trajectory plot saved to: {output_path}")
            plt.close(fig)
        else:
            plt.show()

    def plot_segment_analysis(
        self,
        rally_analyses: List[SegmentAnalysis],
        non_rally_analyses: List[SegmentAnalysis],
        save: bool = True,
    ):
        """
        Plot segment-level analysis comparing individual rallies and non-rallies.

        Args:
            rally_analyses: List of rally segment analyses
            non_rally_analyses: List of non-rally segment analyses
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=self.viz_config["figure_size_large"])

        # Duration comparison
        ax1 = axes[0, 0]
        rally_durations = [seg.duration for seg in rally_analyses]
        non_rally_durations = [seg.duration for seg in non_rally_analyses]
        bp = ax1.boxplot(
            [rally_durations, non_rally_durations],
            labels=["Rally", "Non-Rally"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor(self.viz_config["colors"]["rally"])
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(self.viz_config["colors"]["non_rally"])
        bp["boxes"][1].set_alpha(0.6)
        ax1.set_ylabel("Duration (frames)", fontsize=11, fontweight="bold")
        ax1.set_title("Segment Duration", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="y")

        # Mean velocity comparison
        ax2 = axes[0, 1]
        rally_vel = [seg.mean_velocity for seg in rally_analyses]
        non_rally_vel = [seg.mean_velocity for seg in non_rally_analyses]
        bp = ax2.boxplot(
            [rally_vel, non_rally_vel],
            labels=["Rally", "Non-Rally"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor(self.viz_config["colors"]["rally"])
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(self.viz_config["colors"]["non_rally"])
        bp["boxes"][1].set_alpha(0.6)
        ax2.set_ylabel("Mean Velocity (px/frame)", fontsize=11, fontweight="bold")
        ax2.set_title("Segment Mean Velocity", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        # Trajectory range comparison
        ax3 = axes[1, 0]
        rally_range = [seg.trajectory_range for seg in rally_analyses]
        non_rally_range = [seg.trajectory_range for seg in non_rally_analyses]
        bp = ax3.boxplot(
            [rally_range, non_rally_range],
            labels=["Rally", "Non-Rally"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor(self.viz_config["colors"]["rally"])
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(self.viz_config["colors"]["non_rally"])
        bp["boxes"][1].set_alpha(0.6)
        ax3.set_ylabel("Trajectory Range (px)", fontsize=11, fontweight="bold")
        ax3.set_title("Segment Trajectory Range", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3, axis="y")

        # Large movement percentage
        ax4 = axes[1, 1]
        rally_large = [seg.large_movement_percentage for seg in rally_analyses]
        non_rally_large = [seg.large_movement_percentage for seg in non_rally_analyses]
        bp = ax4.boxplot(
            [rally_large, non_rally_large],
            labels=["Rally", "Non-Rally"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor(self.viz_config["colors"]["rally"])
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(self.viz_config["colors"]["non_rally"])
        bp["boxes"][1].set_alpha(0.6)
        ax4.set_ylabel("Large Movements (%)", fontsize=11, fontweight="bold")
        ax4.set_title("Large Movement Percentage", fontsize=12, fontweight="bold")
        ax4.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "segment_analysis.png"
            plt.savefig(output_path, dpi=self.viz_config["dpi"], bbox_inches="tight")
            print(f"Segment analysis plot saved to: {output_path}")
            plt.close(fig)
        else:
            plt.show()

    # ============================================================================
    # Report Generation
    # ============================================================================

    def print_metrics_comparison(
        self, rally_metrics: StateMetrics, non_rally_metrics: StateMetrics
    ):
        """
        Print detailed comparison of rally vs non-rally metrics.

        Args:
            rally_metrics: Metrics for rally state
            non_rally_metrics: Metrics for non-rally state
        """
        print("\n" + "=" * 80)
        print("RALLY STATE ANALYSIS - METRICS COMPARISON")
        print("=" * 80)

        print(f"\nVideo: {self.video_name}")
        print(f"Rally frames: {rally_metrics.total_frames}")
        print(f"Non-rally frames: {non_rally_metrics.total_frames}")

        # Velocity metrics
        if rally_metrics.velocity and non_rally_metrics.velocity:
            print("\n" + "-" * 80)
            print("VELOCITY METRICS")
            print("-" * 80)
            rv = rally_metrics.velocity
            nv = non_rally_metrics.velocity

            print(
                f"\n{'Metric':<30} {'Rally':<15} {'Non-Rally':<15} {'Difference':<15}"
            )
            print("-" * 80)
            print(
                f"{'Mean Abs Velocity (px/fr)':<30} {rv.mean_abs_velocity:<15.2f} "
                f"{nv.mean_abs_velocity:<15.2f} "
                f"{((rv.mean_abs_velocity/nv.mean_abs_velocity - 1)*100):>+.1f}%"
            )
            print(
                f"{'Median Abs Velocity':<30} {rv.median_abs_velocity:<15.2f} "
                f"{nv.median_abs_velocity:<15.2f} "
                f"{((rv.median_abs_velocity/nv.median_abs_velocity - 1)*100):>+.1f}%"
            )
            print(
                f"{'Std Velocity':<30} {rv.std_velocity:<15.2f} "
                f"{nv.std_velocity:<15.2f} "
                f"{((rv.std_velocity/nv.std_velocity - 1)*100):>+.1f}%"
            )
            print(
                f"{'Max Velocity':<30} {rv.max_velocity:<15.2f} "
                f"{nv.max_velocity:<15.2f} "
                f"{((rv.max_velocity/nv.max_velocity - 1)*100):>+.1f}%"
            )
            print(
                f"{'Large Movements %':<30} {rv.large_movement_percentage:<15.2f} "
                f"{nv.large_movement_percentage:<15.2f} "
                f"{(rv.large_movement_percentage - nv.large_movement_percentage):>+.1f}pp"
            )
            print(
                f"{'Stationary Frames %':<30} {rv.stationary_frame_percentage:<15.2f} "
                f"{nv.stationary_frame_percentage:<15.2f} "
                f"{(rv.stationary_frame_percentage - nv.stationary_frame_percentage):>+.1f}pp"
            )

        # Position metrics
        if rally_metrics.position and non_rally_metrics.position:
            print("\n" + "-" * 80)
            print("POSITION METRICS")
            print("-" * 80)
            rp = rally_metrics.position
            np_met = non_rally_metrics.position

            print(
                f"\n{'Metric':<30} {'Rally':<15} {'Non-Rally':<15} {'Difference':<15}"
            )
            print("-" * 80)
            print(
                f"{'Trajectory Range (px)':<30} {rp.trajectory_range:<15.1f} "
                f"{np_met.trajectory_range:<15.1f} "
                f"{((rp.trajectory_range/np_met.trajectory_range - 1)*100):>+.1f}%"
            )
            print(
                f"{'Std Position':<30} {rp.std_position:<15.1f} "
                f"{np_met.std_position:<15.1f} "
                f"{((rp.std_position/np_met.std_position - 1)*100):>+.1f}%"
            )
            print(
                f"{'Mean Abs Deviation':<30} {rp.mean_abs_deviation:<15.1f} "
                f"{np_met.mean_abs_deviation:<15.1f} "
                f"{((rp.mean_abs_deviation/np_met.mean_abs_deviation - 1)*100):>+.1f}%"
            )

        # Motion pattern metrics
        if rally_metrics.motion and non_rally_metrics.motion:
            print("\n" + "-" * 80)
            print("MOTION PATTERN METRICS")
            print("-" * 80)
            rm = rally_metrics.motion
            nm = non_rally_metrics.motion

            print(
                f"\n{'Metric':<30} {'Rally':<15} {'Non-Rally':<15} {'Difference':<15}"
            )
            print("-" * 80)
            print(
                f"{'Mean Abs Acceleration':<30} {rm.mean_abs_acceleration:<15.2f} "
                f"{nm.mean_abs_acceleration:<15.2f} "
                f"{((rm.mean_abs_acceleration/nm.mean_abs_acceleration - 1)*100):>+.1f}%"
            )
            print(
                f"{'Coefficient of Variation':<30} {rm.coefficient_of_variation:<15.2f} "
                f"{nm.coefficient_of_variation:<15.2f} "
                f"{((rm.coefficient_of_variation/nm.coefficient_of_variation - 1)*100):>+.1f}%"
            )
            print(
                f"{'High Acceleration %':<30} {rm.high_acceleration_percentage:<15.2f} "
                f"{nm.high_acceleration_percentage:<15.2f} "
                f"{(rm.high_acceleration_percentage - nm.high_acceleration_percentage):>+.1f}pp"
            )

            print(f"\nVelocity Autocorrelation:")
            for lag in self.stats_config["autocorrelation_lags"]:
                r_auto = rm.velocity_autocorrelations[lag]
                n_auto = nm.velocity_autocorrelations[lag]
                print(
                    f"  Lag {lag:<2}: Rally = {r_auto:>6.3f}, "
                    f"Non-Rally = {n_auto:>6.3f}"
                )

        # Oscillation metrics
        if rally_metrics.oscillation and non_rally_metrics.oscillation:
            print("\n" + "-" * 80)
            print("OSCILLATION METRICS")
            print("-" * 80)
            ro = rally_metrics.oscillation
            no = non_rally_metrics.oscillation

            print(
                f"\n{'Metric':<30} {'Rally':<15} {'Non-Rally':<15} {'Difference':<15}"
            )
            print("-" * 80)
            print(
                f"{'Direction Change Rate %':<30} {ro.direction_change_rate:<15.2f} "
                f"{no.direction_change_rate:<15.2f} "
                f"{(ro.direction_change_rate - no.direction_change_rate):>+.1f}pp"
            )
            print(
                f"{'Mean Oscillation Amp (px)':<30} {ro.mean_oscillation_amplitude:<15.1f} "
                f"{no.mean_oscillation_amplitude:<15.1f} "
                f"{((ro.mean_oscillation_amplitude/no.mean_oscillation_amplitude - 1)*100 if no.mean_oscillation_amplitude > 0 else 0):>+.1f}%"
            )
            print(
                f"{'Time Between Changes (fr)':<30} {ro.mean_time_between_changes:<15.1f} "
                f"{no.mean_time_between_changes:<15.1f} "
                f"{((ro.mean_time_between_changes/no.mean_time_between_changes - 1)*100 if no.mean_time_between_changes > 0 else 0):>+.1f}%"
            )

        print("\n" + "=" * 80)

    def generate_report(self) -> Dict:
        """
        Generate comprehensive analysis report.

        Returns:
            Dictionary with all analysis results
        """
        print("\n" + "=" * 80)
        print("RALLY STATE ANALYZER - COMPREHENSIVE REPORT")
        print("=" * 80)

        # Load data
        df = self.load_annotations()
        label_col = self.annotation_config["label_column"]
        rally_mask = df[label_col] == "start"
        non_rally_mask = ~rally_mask

        # Compute metrics for each state
        print("\nComputing metrics for rally state...")
        rally_metrics = self.analyze_state("Rally", rally_mask, df)

        print("Computing metrics for non-rally state...")
        non_rally_metrics = self.analyze_state("Non-Rally", non_rally_mask, df)

        # Analyze individual segments
        print("Analyzing individual segments...")
        rally_analyses, non_rally_analyses = self.analyze_segments(df)

        # Print comparison
        self.print_metrics_comparison(rally_metrics, non_rally_metrics)

        # Print segment summaries
        if self.compute_metrics_flags["temporal"]:
            print("\n" + "-" * 80)
            print("TEMPORAL ANALYSIS")
            print("-" * 80)
            print(f"\nRally Segments: {len(rally_analyses)}")
            rally_durations = [seg.duration for seg in rally_analyses]
            print(f"  Mean duration: {np.mean(rally_durations):.1f} frames")
            print(f"  Median duration: {np.median(rally_durations):.1f} frames")
            print(
                f"  Range: {np.min(rally_durations):.0f} - {np.max(rally_durations):.0f} frames"
            )

            print(f"\nNon-Rally Segments: {len(non_rally_analyses)}")
            non_rally_durations = [seg.duration for seg in non_rally_analyses]
            print(f"  Mean duration: {np.mean(non_rally_durations):.1f} frames")
            print(f"  Median duration: {np.median(non_rally_durations):.1f} frames")
            print(
                f"  Range: {np.min(non_rally_durations):.0f} - {np.max(non_rally_durations):.0f} frames"
            )

        # Generate visualizations
        print("\n" + "-" * 80)
        print("GENERATING VISUALIZATIONS")
        print("-" * 80)

        if self.compute_metrics_flags["velocity"]:
            print("Creating velocity distribution plots...")
            self.plot_velocity_distributions(rally_metrics, non_rally_metrics)

        if self.compute_metrics_flags["position"]:
            print("Creating position distribution plots...")
            self.plot_position_distributions(rally_metrics, non_rally_metrics)

        print("Creating trajectory plot...")
        self.plot_trajectory_with_states()

        if self.compute_metrics_flags["temporal"]:
            print("Creating segment analysis plots...")
            self.plot_segment_analysis(rally_analyses, non_rally_analyses)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print(f"All outputs saved to: {self.output_dir}")
        print("=" * 80)

        return {
            "rally_metrics": rally_metrics,
            "non_rally_metrics": non_rally_metrics,
            "rally_segment_analyses": rally_analyses,
            "non_rally_segment_analyses": non_rally_analyses,
        }


def main():
    """Main entry point for ground truth analysis."""
    analyzer = RallyStateAnalyzer()
    report = analyzer.generate_report()

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    rally_m = report["rally_metrics"]
    non_rally_m = report["non_rally_metrics"]

    if rally_m.velocity and non_rally_m.velocity:
        vel_diff = (
            rally_m.velocity.mean_abs_velocity / non_rally_m.velocity.mean_abs_velocity
            - 1
        ) * 100
        print(
            f"\nVelocity: Rally periods have {vel_diff:.1f}% higher mean velocity than non-rally"
        )

    if rally_m.position and non_rally_m.position:
        range_diff = (
            rally_m.position.trajectory_range / non_rally_m.position.trajectory_range
            - 1
        ) * 100
        print(f"Position: Rally periods cover {range_diff:.1f}% more vertical space")

    if rally_m.motion and non_rally_m.motion:
        auto_diff = (
            rally_m.motion.velocity_autocorrelations[1]
            - non_rally_m.motion.velocity_autocorrelations[1]
        )
        print(
            f"Motion: Rally periods show {auto_diff:.2f} higher velocity autocorrelation "
            f"(more sustained motion)"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
