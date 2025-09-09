import json
import os
import matplotlib.pyplot as plt
from typing import Dict


class SquashDetectionVisualizer:
    def __init__(self, annotation_file: str, output_dir: str = "plots"):
        """
        Initialize the visualizer.

        Args:
            annotation_file: Path to the annotations_with_detections.json file
            output_dir: Directory where to save the plots
        """
        self.annotation_file = annotation_file
        self.output_dir = output_dir
        self.annotations = self._load_annotations()

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def _load_annotations(self) -> Dict:
        """Load the annotations from the JSON file."""
        with open(self.annotation_file, "r") as f:
            return json.load(f)

    def plot_all_time_series(self):
        """
        Create separate time series plots for ball and player positions for each video.
        """
        for video_name, annotations in self.annotations.items():
            print(f"Processing video: {video_name}")

            # Create separate plots for each video
            self._plot_ball_time_series(video_name, annotations)
            self._plot_players_time_series(video_name, annotations)

    def _plot_ball_time_series(self, video_name: str, annotations: list):
        """Create time series plot for ball positions."""
        # Create figure for ball position
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Ball Position over Time - {video_name}", fontsize=16)

        # Prepare data for each annotation segment
        for annotation in annotations:
            label = annotation["label"]
            detections = annotation["detections"]

            # Convert string keys to integers and sort by frame number
            frames = sorted([int(k) for k in detections.keys()])

            # Extract ball positions
            ball_frames = []
            ball_x = []
            ball_y = []

            for frame in frames:
                frame_data = detections[str(frame)]
                ball_pos = frame_data.get("ball", None)

                if ball_pos and len(ball_pos) >= 2:
                    ball_frames.append(frame)
                    ball_x.append(ball_pos[0])
                    ball_y.append(ball_pos[1])

            # Skip if no ball data
            if not ball_frames:
                continue

            # Plot X position
            if label == "rally_start":
                color = "g"
                marker = "o"
            elif label == "in_play":
                color = "b"
                marker = "s"
            else:  # rally_end
                color = "r"
                marker = "^"

            ax1.plot(
                ball_frames,
                ball_x,
                color=color,
                marker=marker,
                linestyle="-",
                label=f"{label}",
                alpha=0.7,
                markersize=4,
            )

            # Plot Y position
            ax2.plot(
                ball_frames,
                ball_y,
                color=color,
                marker=marker,
                linestyle="-",
                label=f"{label}",
                alpha=0.7,
                markersize=4,
            )

        # Set plot labels and formatting
        ax1.set_title("Ball X Position")
        ax1.set_ylabel("X Position")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.set_title("Ball Y Position")
        ax2.set_xlabel("Frame Number")
        ax2.set_ylabel("Y Position")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Invert Y axis (since origin is at top-left)
        ax2.invert_yaxis()

        plt.tight_layout()

        # Save the plot
        output_file = os.path.join(
            self.output_dir, f"{video_name.replace('.mp4', '')}_ball_time_series.png"
        )
        plt.savefig(output_file, dpi=300)
        print(f"Saved ball time series plot to {output_file}")
        plt.close()

    def _plot_players_time_series(self, video_name: str, annotations: list):
        """Create time series plot for player positions."""
        # Create figure for player positions
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Player Positions over Time - {video_name}", fontsize=16)

        # Prepare data for each annotation segment
        for annotation in annotations:
            label = annotation["label"]
            detections = annotation["detections"]

            # Convert string keys to integers and sort by frame number
            frames = sorted([int(k) for k in detections.keys()])

            # Extract player positions
            p1_frames = []
            p1_x = []
            p1_y = []

            p2_frames = []
            p2_x = []
            p2_y = []

            for frame in frames:
                frame_data = detections[str(frame)]
                players_pos = frame_data.get("players_position", [])

                # Player 1
                if len(players_pos) > 0 and players_pos[0] and len(players_pos[0]) >= 2:
                    p1_frames.append(frame)
                    p1_x.append(players_pos[0][0])
                    p1_y.append(players_pos[0][1])

                # Player 2
                if len(players_pos) > 1 and players_pos[1] and len(players_pos[1]) >= 2:
                    p2_frames.append(frame)
                    p2_x.append(players_pos[1][0])
                    p2_y.append(players_pos[1][1])

            # Skip if no player data
            if not p1_frames and not p2_frames:
                continue

            # Set line style based on label
            if label == "rally_start":
                linestyle = "-"
            elif label == "in_play":
                linestyle = "--"
            else:  # rally_end
                linestyle = ":"

            # Plot Player 1 X position
            if p1_frames:
                ax1.plot(
                    p1_frames,
                    p1_x,
                    "b",
                    marker="o",
                    linestyle=linestyle,
                    label=f"Player 1 - {label}",
                    alpha=0.7,
                    markersize=4,
                )

            # Plot Player 2 X position
            if p2_frames:
                ax1.plot(
                    p2_frames,
                    p2_x,
                    "r",
                    marker="s",
                    linestyle=linestyle,
                    label=f"Player 2 - {label}",
                    alpha=0.7,
                    markersize=4,
                )

            # Plot Player 1 Y position
            if p1_frames:
                ax2.plot(
                    p1_frames,
                    p1_y,
                    "b",
                    marker="o",
                    linestyle=linestyle,
                    label=f"Player 1 - {label}",
                    alpha=0.7,
                    markersize=4,
                )

            # Plot Player 2 Y position
            if p2_frames:
                ax2.plot(
                    p2_frames,
                    p2_y,
                    "r",
                    marker="s",
                    linestyle=linestyle,
                    label=f"Player 2 - {label}",
                    alpha=0.7,
                    markersize=4,
                )

        # Set plot labels and formatting
        ax1.set_title("Player X Positions")
        ax1.set_ylabel("X Position")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.set_title("Player Y Positions")
        ax2.set_xlabel("Frame Number")
        ax2.set_ylabel("Y Position")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Invert Y axis (since origin is at top-left)
        ax2.invert_yaxis()

        plt.tight_layout()

        # Save the plot
        output_file = os.path.join(
            self.output_dir, f"{video_name.replace('.mp4', '')}_players_time_series.png"
        )
        plt.savefig(output_file, dpi=300)
        print(f"Saved player time series plot to {output_file}")
        plt.close()

    def plot_separated_player_time_series(self):
        """Create separate time series plots for each player."""
        for video_name, annotations in self.annotations.items():
            print(f"Processing individual player plots for: {video_name}")

            # Create separate plots for each player
            for player_idx in [
                0,
                1,
            ]:  # Player indices 0 and 1 (for Player 1 and Player 2)
                self._plot_single_player_time_series(
                    video_name, annotations, player_idx
                )

    def _plot_single_player_time_series(
        self, video_name: str, annotations: list, player_idx: int
    ):
        """Create time series plot for a single player."""
        player_num = player_idx + 1  # Convert to 1-indexed player numbers

        # Create figure for this player
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(
            f"Player {player_num} Position over Time - {video_name}", fontsize=16
        )

        # Prepare data for each annotation segment
        for annotation in annotations:
            label = annotation["label"]
            detections = annotation["detections"]

            # Convert string keys to integers and sort by frame number
            frames = sorted([int(k) for k in detections.keys()])

            # Extract player positions
            p_frames = []
            p_x = []
            p_y = []

            for frame in frames:
                frame_data = detections[str(frame)]
                players_pos = frame_data.get("players_position", [])

                if (
                    len(players_pos) > player_idx
                    and players_pos[player_idx]
                    and len(players_pos[player_idx]) >= 2
                ):
                    p_frames.append(frame)
                    p_x.append(players_pos[player_idx][0])
                    p_y.append(players_pos[player_idx][1])

            # Skip if no data
            if not p_frames:
                continue

            # Set color and line style based on label
            if label == "rally_start":
                color = "g"
                linestyle = "-"
            elif label == "in_play":
                color = "b"
                linestyle = "--"
            else:  # rally_end
                color = "r"
                linestyle = ":"

            # Plot X position
            ax1.plot(
                p_frames,
                p_x,
                color=color,
                marker="o",
                linestyle=linestyle,
                label=f"{label}",
                alpha=0.7,
                markersize=4,
            )

            # Plot Y position
            ax2.plot(
                p_frames,
                p_y,
                color=color,
                marker="o",
                linestyle=linestyle,
                label=f"{label}",
                alpha=0.7,
                markersize=4,
            )

        # Set plot labels and formatting
        ax1.set_title(f"Player {player_num} X Position")
        ax1.set_ylabel("X Position")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.set_title(f"Player {player_num} Y Position")
        ax2.set_xlabel("Frame Number")
        ax2.set_ylabel("Y Position")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Invert Y axis (since origin is at top-left)
        ax2.invert_yaxis()

        plt.tight_layout()

        # Save the plot
        output_file = os.path.join(
            self.output_dir,
            f"{video_name.replace('.mp4', '')}_player{player_num}_time_series.png",
        )
        plt.savefig(output_file, dpi=300)
        print(f"Saved Player {player_num} time series plot to {output_file}")
        plt.close()


# Example usage
if __name__ == "__main__":
    visualizer = SquashDetectionVisualizer(
        annotation_file="annotations_with_detections.json", output_dir="squash_plots"
    )

    # Create time series plots for ball and players
    visualizer.plot_all_time_series()

    # Optionally, create separate plots for each player
    visualizer.plot_separated_player_time_series()
