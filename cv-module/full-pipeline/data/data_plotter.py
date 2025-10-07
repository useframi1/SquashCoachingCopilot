import os
import seaborn as sns
import matplotlib.pyplot as plt


class DataPlotter:
    """
    Generates and saves visual comparisons of raw vs postprocessed data.
    """

    def __init__(self, output_dir="output/plots"):
        self.output_dir = output_dir
        sns.set_theme(style="whitegrid", context="talk")

    def compare_before_after(self, raw_frames, processed_frames):
        """
        Generate and save plots for both raw and postprocessed data.
        """
        os.makedirs(os.path.join(self.output_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "postprocessed"), exist_ok=True)

        print("📈 Generating plots for raw data...")
        self._plot_all(raw_frames, os.path.join(self.output_dir, "raw"))

        print("📊 Generating plots for postprocessed data...")
        self._plot_all(processed_frames, os.path.join(self.output_dir, "postprocessed"))

        print(f"✅ All plots saved under '{self.output_dir}/'")

    # ----------------------------
    # Internal plotting functions
    # ----------------------------
    def _plot_all(self, frames, save_dir):
        self._plot_ball_trajectory(frames, save_dir)
        self._plot_player_positions(frames, save_dir)
        self._plot_rally_states(frames, save_dir)
        self._plot_stroke_timeline(frames, save_dir)

    def _plot_ball_trajectory(self, frames, save_dir):
        timestamps, xs, ys = [], [], []
        for f in frames:
            if f.ball and f.ball.position:
                timestamps.append(f.timestamp)
                xs.append(f.ball.position[0])
                ys.append(f.ball.position[1])

        if not timestamps:
            print("⚠️ No ball data found, skipping ball trajectory plot.")
            return

        # Create figure with two subplots (x and y coordinates)
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot X coordinates over time
        ax1.plot(timestamps, xs, marker="o", markersize=3, linewidth=1)
        ax1.set_title("Ball X Coordinate Over Time")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("X Position (pixels)")
        ax1.grid(True)

        # Plot Y coordinates over time
        ax2.plot(timestamps, ys, marker="o", markersize=3, linewidth=1)
        ax2.set_title("Ball Y Coordinate Over Time")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Y Position (pixels)")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "ball_trajectory.png"))
        plt.close()

    def _plot_player_positions(self, frames, save_dir):
        timestamps = []
        p1x, p1y, p2x, p2y = [], [], [], []
        for f in frames:
            timestamps.append(f.timestamp)
            if f.player1 and f.player1.real_position:
                p1x.append(f.player1.real_position[0])
                p1y.append(f.player1.real_position[1])
            else:
                p1x.append(None)
                p1y.append(None)
            if f.player2 and f.player2.real_position:
                p2x.append(f.player2.real_position[0])
                p2y.append(f.player2.real_position[1])
            else:
                p2x.append(None)
                p2y.append(None)

        # Create figure with two subplots (x and y coordinates)
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot X coordinates over time
        ax1.plot(timestamps, p1x, label="Player 1", linewidth=2, marker="o", markersize=3)
        ax1.plot(timestamps, p2x, label="Player 2", linewidth=2, marker="o", markersize=3)
        ax1.set_title("Player X Coordinates Over Time")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("X Position (pixels)")
        ax1.legend()
        ax1.grid(True)

        # Plot Y coordinates over time
        ax2.plot(timestamps, p1y, label="Player 1", linewidth=2, marker="o", markersize=3)
        ax2.plot(timestamps, p2y, label="Player 2", linewidth=2, marker="o", markersize=3)
        ax2.set_title("Player Y Coordinates Over Time")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Y Position (pixels)")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "player_positions.png"))
        plt.close()

    def _plot_rally_states(self, frames, save_dir):
        timestamps = [f.timestamp for f in frames]
        states = [f.rally_state for f in frames]
        state_map = {"start": 0, "active": 1, "end": 2}
        y = [state_map.get(s, None) for s in states]

        plt.figure(figsize=(12, 3))
        plt.step(timestamps, y, where="post", linewidth=2)
        plt.yticks(list(state_map.values()), list(state_map.keys()))
        plt.title("Rally States Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("State")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "rally_states.png"))
        plt.close()

    def _plot_stroke_timeline(self, frames, save_dir):
        timestamps = [f.timestamp for f in frames]
        p1_strokes = [f.player1.stroke_type if f.player1 else None for f in frames]
        p2_strokes = [f.player2.stroke_type if f.player2 else None for f in frames]

        stroke_map = {"neither": 0, "forehand": 1, "backhand": 2}

        p1_vals = [stroke_map.get(s, None) for s in p1_strokes]
        p2_vals = [stroke_map.get(s, None) for s in p2_strokes]

        # Create figure with two subplots (one for each player)
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot Player 1 strokes
        ax1.step(timestamps, p1_vals, where="post", linewidth=2)
        ax1.set_yticks(list(stroke_map.values()))
        ax1.set_yticklabels(list(stroke_map.keys()))
        ax1.set_title("Player 1 Stroke Types Over Time")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Stroke Type")
        ax1.grid(True)

        # Plot Player 2 strokes
        ax2.step(timestamps, p2_vals, where="post", linewidth=2)
        ax2.set_yticks(list(stroke_map.values()))
        ax2.set_yticklabels(list(stroke_map.keys()))
        ax2.set_title("Player 2 Stroke Types Over Time")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Stroke Type")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "stroke_timeline.png"))
        plt.close()
