import numpy as np


class PlayerAnalyzer:
    """Analyzes player movements and positions for intensity and position tracking"""

    def __init__(self, config):
        self.config = config
        processing_config = config.get("processing", {})

        self.interval = processing_config.get("average_position_interval", 30)
        self.max_history = processing_config.get("max_position_history", 100)

        # Player tracking data
        self.player1_positions = []
        self.player2_positions = []
        self.player1_movements = []
        self.player2_movements = []
        self.player1_last_pos = None
        self.player2_last_pos = None

        # Current averages
        self.current_p1_avg = None
        self.current_p2_avg = None
        self.current_p1_intensity = None
        self.current_p2_intensity = None

        # Distance tracking
        self.player_distances = []
        self.current_avg_distance = None

    def update_positions(self, real_coords):
        """Update player positions and calculate movement intensity"""
        # Update Player 1
        if 1 in real_coords and len(real_coords[1]) > 0:
            current_pos = real_coords[1][-1]
            self.player1_positions.append(current_pos)

            # Calculate movement intensity
            if self.player1_last_pos is not None:
                distance = np.sqrt(
                    (current_pos[0] - self.player1_last_pos[0]) ** 2
                    + (current_pos[1] - self.player1_last_pos[1]) ** 2
                )
                self.player1_movements.append(distance)

            self.player1_last_pos = current_pos

            # Keep history limited
            if len(self.player1_positions) > self.max_history:
                self.player1_positions.pop(0)
            if len(self.player1_movements) > self.max_history:
                self.player1_movements.pop(0)

        # Update Player 2
        if 2 in real_coords and len(real_coords[2]) > 0:
            current_pos = real_coords[2][-1]
            self.player2_positions.append(current_pos)

            # Calculate movement intensity
            if self.player2_last_pos is not None:
                distance = np.sqrt(
                    (current_pos[0] - self.player2_last_pos[0]) ** 2
                    + (current_pos[1] - self.player2_last_pos[1]) ** 2
                )
                self.player2_movements.append(distance)

            self.player2_last_pos = current_pos

            # Keep history limited
            if len(self.player2_positions) > self.max_history:
                self.player2_positions.pop(0)
            if len(self.player2_movements) > self.max_history:
                self.player2_movements.pop(0)

        # Calculate distance between players if both are available
        if len(self.player1_positions) > 0 and len(self.player2_positions) > 0:
            p1_pos = self.player1_positions[-1]
            p2_pos = self.player2_positions[-1]
            distance = np.sqrt(
                (p1_pos[0] - p2_pos[0]) ** 2 + (p1_pos[1] - p2_pos[1]) ** 2
            )
            self.player_distances.append(distance)

            # Keep distance history limited
            if len(self.player_distances) > self.max_history:
                self.player_distances.pop(0)

    def calculate_averages(self, frame_count):
        """Calculate average positions and intensities over the interval"""
        should_update = frame_count % self.interval == 0

        if should_update:
            # Calculate Player 1 averages
            if self.player1_positions:
                recent_positions = self.player1_positions[-self.interval :]
                avg_x = np.mean([pos[0] for pos in recent_positions])
                avg_y = np.mean([pos[1] for pos in recent_positions])
                self.current_p1_avg = (avg_x, avg_y)

                if self.player1_movements:
                    recent_movements = self.player1_movements[-(self.interval - 1) :]
                    self.current_p1_intensity = np.mean(recent_movements)
                else:
                    self.current_p1_intensity = 0.0
            else:
                self.current_p1_avg = None
                self.current_p1_intensity = None

            # Calculate Player 2 averages
            if self.player2_positions:
                recent_positions = self.player2_positions[-self.interval :]
                avg_x = np.mean([pos[0] for pos in recent_positions])
                avg_y = np.mean([pos[1] for pos in recent_positions])
                self.current_p2_avg = (avg_x, avg_y)

                if self.player2_movements:
                    recent_movements = self.player2_movements[-(self.interval - 1) :]
                    self.current_p2_intensity = np.mean(recent_movements)
                else:
                    self.current_p2_intensity = 0.0
            else:
                self.current_p2_avg = None
                self.current_p2_intensity = None

            # Calculate average distance between players
            if self.player_distances:
                recent_distances = self.player_distances[-self.interval :]
                self.current_avg_distance = np.mean(recent_distances)
            else:
                self.current_avg_distance = None

        return should_update

    def get_combined_intensity(self):
        """Calculate combined intensity from both players"""
        if (
            self.current_p1_intensity is not None
            and self.current_p2_intensity is not None
        ):
            return (self.current_p1_intensity + self.current_p2_intensity) / 2
        elif self.current_p1_intensity is not None:
            return self.current_p1_intensity
        elif self.current_p2_intensity is not None:
            return self.current_p2_intensity
        return 0.0

    def get_player_stats(self):
        """Get current player statistics"""
        return {
            "player1": {
                "avg_position": self.current_p1_avg,
                "intensity": self.current_p1_intensity,
                "total_positions": len(self.player1_positions),
            },
            "player2": {
                "avg_position": self.current_p2_avg,
                "intensity": self.current_p2_intensity,
                "total_positions": len(self.player2_positions),
            },
            "combined_intensity": self.get_combined_intensity(),
            "avg_distance": self.current_avg_distance,
            "total_distance_measurements": len(self.player_distances),
        }

    def print_stats(self, frame_count):
        """Print player statistics to console"""
        stats = self.get_player_stats()

        print(f"\n--- Frame {frame_count} ---")

        # Player 1
        if stats["player1"]["avg_position"] is not None:
            pos = stats["player1"]["avg_position"]
            intensity = stats["player1"]["intensity"]
            total = stats["player1"]["total_positions"]
            print(
                f"Player 1 - Avg pos: ({pos[0]:.2f}, {pos[1]:.2f})m, "
                f"Avg intensity: {intensity:.3f}m/frame (total: {total})"
            )
        else:
            print("Player 1: No positions detected in recent frames")

        # Player 2
        if stats["player2"]["avg_position"] is not None:
            pos = stats["player2"]["avg_position"]
            intensity = stats["player2"]["intensity"]
            total = stats["player2"]["total_positions"]
            print(
                f"Player 2 - Avg pos: ({pos[0]:.2f}, {pos[1]:.2f})m, "
                f"Avg intensity: {intensity:.3f}m/frame (total: {total})"
            )
        else:
            print("Player 2: No positions detected in recent frames")

        print(f"Combined Intensity: {stats['combined_intensity']:.3f} m/frame")

        # Average distance between players
        if stats["avg_distance"] is not None:
            print(f"Average Distance Between Players: {stats['avg_distance']:.2f} m")
        else:
            print("Average Distance: No data available")
