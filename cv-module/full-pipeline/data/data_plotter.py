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

        plt.figure(figsize=(10, 6))
        plt.plot(xs, ys, marker="o", markersize=3, linewidth=1)
        plt.title("Ball Trajectory (X-Y Path)")
        plt.xlabel("X Position (pixels)")
        plt.ylabel("Y Position (pixels)")
        plt.grid(True)
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

        plt.figure(figsize=(10, 6))
        sns.lineplot(x=p1x, y=p1y, label="Player 1", linewidth=2)
        sns.lineplot(x=p2x, y=p2y, label="Player 2", linewidth=2)
        plt.title("Player Trajectories on Court")
        plt.xlabel("X Position (pixels)")
        plt.ylabel("Y Position (pixels)")
        plt.legend()
        plt.grid(True)
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

        plt.figure(figsize=(12, 4))
        plt.scatter(timestamps, p1_vals, s=30, label="Player 1", alpha=0.7)
        plt.scatter(timestamps, p2_vals, s=30, label="Player 2", alpha=0.7)
        plt.yticks(list(stroke_map.values()), list(stroke_map.keys()))
        plt.title("Stroke Types Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Stroke Type")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "stroke_timeline.png"))
        plt.close()
