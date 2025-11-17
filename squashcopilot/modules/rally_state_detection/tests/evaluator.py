"""
Rally State Detection Evaluator

Visualizes rally state annotations by plotting ball trajectory with state transitions.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter

from squashcopilot.common.utils import load_config


class RallyStateEvaluator:
    """Evaluates and visualizes rally state annotations."""

    def __init__(self, config_name: str = "rally_state_detection"):
        """
        Initialize evaluator with configuration.

        Args:
            config_name: Name of the configuration file (without .yaml extension)
        """
        full_config = load_config(config_name=config_name)
        self.config = full_config["evaluator"]
        self.annotation_config = full_config["annotation"]
        self.video_name = self.config["video_name"]

        # Get project root and build paths
        project_root = Path(__file__).parent.parent.parent.parent.parent

        # Data directory (where annotations are saved)
        data_base_dir = project_root / self.config["data_dir"]
        self.data_dir = data_base_dir / self.video_name

        # Output directory (where plots will be saved)
        output_base_dir = project_root / self.config["output_dir"]
        self.output_dir = output_base_dir / self.video_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_annotations(self) -> pd.DataFrame:
        """
        Load rally state annotations from CSV file.

        Returns:
            DataFrame with frame, features, and rally state labels
        """
        csv_path = self.data_dir / f"{self.video_name}_annotations.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} annotated frames from {csv_path}")

        return df

    def preprocess_ball_trajectory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess ball trajectory using Savitzky-Golay filter.

        Args:
            df: DataFrame with ball_y column

        Returns:
            DataFrame with additional ball_y_filtered column
        """
        # Apply Savitzky-Golay filter to smooth the trajectory
        # window_length must be odd and greater than polyorder
        window_length = 11  # Can be adjusted for more/less smoothing
        polyorder = 3  # Polynomial order

        # Ensure we have enough data points
        if len(df) < window_length:
            print(
                f"Warning: Not enough data points for filtering. Using original values."
            )
            df["ball_y_filtered"] = df["ball_y"]
        else:
            df["ball_y_filtered"] = savgol_filter(
                df["ball_y"], window_length=window_length, polyorder=polyorder
            )
            print(
                f"Applied Savitzky-Golay filter (window={window_length}, poly={polyorder})"
            )

        return df

    def plot_trajectory(self, df: pd.DataFrame):
        """
        Plot ball y trajectory with rally state transitions.

        Args:
            df: DataFrame with annotations (must have ball_y_filtered column)
        """
        # Get label column name from annotation config
        label_col = self.annotation_config["label_column"]

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))

        # Plot original ball y trajectory (lighter, thinner)
        ax.plot(
            df["frame"],
            df["ball_y"],
            linewidth=1,
            color="lightblue",
            alpha=0.5,
            label="Ball Y Position (Original)",
        )

        # Plot filtered ball y trajectory (darker, thicker)
        ax.plot(
            df["frame"],
            df["ball_y_filtered"],
            linewidth=2.5,
            color="blue",
            label="Ball Y Position (Filtered)",
        )

        # Find state transitions
        state_changes = []
        for i in range(1, len(df)):
            if df[label_col].iloc[i] != df[label_col].iloc[i - 1]:
                state_changes.append(i)

        # Add vertical lines at state transitions
        for change_idx in state_changes:
            frame_num = df["frame"].iloc[change_idx]
            ax.axvline(x=frame_num, color="red", linestyle="--", linewidth=2, alpha=0.7)

        # Color the background based on rally state
        start_color = (0.0, 1.0, 0.0, 0.2)  # Green with alpha
        end_color = (1.0, 0.0, 0.0, 0.2)  # Red with alpha

        current_state = df[label_col].iloc[0]
        region_start = df["frame"].iloc[0]

        for i in range(1, len(df)):
            if df[label_col].iloc[i] != current_state or i == len(df) - 1:
                region_end = (
                    df["frame"].iloc[i] if i == len(df) - 1 else df["frame"].iloc[i - 1]
                )

                # Fill region with appropriate color
                color = start_color if current_state == "start" else end_color
                ax.axvspan(
                    region_start,
                    region_end,
                    alpha=0.3,
                    color=color[0:3],
                    label=(
                        f"{current_state.capitalize()}"
                        if region_start == df["frame"].iloc[0]
                        else ""
                    ),
                )

                if i < len(df):
                    current_state = df[label_col].iloc[i]
                    region_start = df["frame"].iloc[i]

        # Customize plot
        ax.set_xlabel("Frame Number", fontsize=14, fontweight="bold")
        ax.set_ylabel("Ball Y Coordinate", fontsize=14, fontweight="bold")
        ax.set_title(
            f"Rally State Trajectory - {self.video_name}",
            fontsize=16,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        ax.legend(loc="upper right", fontsize=11)

        # Add annotation for state changes
        if state_changes:
            ax.text(
                0.02,
                0.98,
                f"State Transitions: {len(state_changes)}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()

        # Save plot
        output_path = self.output_dir / f"{self.video_name}_trajectory.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Trajectory plot saved to: {output_path}")

    def print_summary(self, df: pd.DataFrame):
        """
        Print summary statistics of the annotations.

        Args:
            df: DataFrame with annotations
        """
        label_col = self.annotation_config["label_column"]

        print("\n" + "=" * 60)
        print("ANNOTATION SUMMARY")
        print("=" * 60)
        print(f"Video: {self.video_name}")
        print(f"Total Frames: {len(df)}")
        print(f"\nState Distribution:")

        state_counts = df[label_col].value_counts()
        for state, count in state_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {state.capitalize()}: {count} frames ({percentage:.2f}%)")

        # Count transitions
        transitions = 0
        for i in range(1, len(df)):
            if df[label_col].iloc[i] != df[label_col].iloc[i - 1]:
                transitions += 1

        print(f"\nState Transitions: {transitions}")
        print("=" * 60)

    def run(self):
        """Run the complete evaluation pipeline."""
        print("=" * 60)
        print("RALLY STATE ANNOTATION EVALUATOR")
        print("=" * 60)

        # Load annotations
        print(f"\nLoading annotations for {self.video_name}...")
        df = self.load_annotations()

        # Preprocess ball trajectory
        print(f"\nPreprocessing ball trajectory...")
        df = self.preprocess_ball_trajectory(df)

        # Print summary
        self.print_summary(df)

        # Plot trajectory
        print(f"\nGenerating trajectory plot...")
        self.plot_trajectory(df)

        print("\n" + "=" * 60)
        print("✓ Evaluation complete!")
        print(f"✓ Output saved to: {self.output_dir}")
        print("=" * 60)


def main():
    """Main entry point."""
    evaluator = RallyStateEvaluator()
    evaluator.run()


if __name__ == "__main__":
    main()
