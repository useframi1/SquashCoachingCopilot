import numpy as np


class RallyStateDetector:
    """Handles rally state detection and transitions for squash games"""

    def __init__(self, config):
        self.config = config
        self.rally_config = config.get("rally_segmentation", {})

        # Load thresholds from config
        self.start_intensity_threshold = self.rally_config.get(
            "start_intensity_threshold", 0.025
        )
        self.active_intensity_threshold = self.rally_config.get(
            "active_intensity_threshold", 0.035
        )
        self.end_intensity_threshold = self.rally_config.get(
            "end_intensity_threshold", 0.025
        )

        # Court dimensions from config
        self.service_line = self.rally_config.get("service_line", 5.44)
        self.service_box_back = self.rally_config.get("service_box_back", 7.04)
        self.court_center_x = self.rally_config.get("court_center_x", 3.2)

        # Service box boundaries from config
        service_boxes = self.rally_config.get(
            "service_boxes",
            {"left": {"x_min": 0, "x_max": 1.6}, "right": {"x_min": 4.8, "x_max": 6.4}},
        )
        self.left_box = service_boxes["left"]
        self.right_box = service_boxes["right"]

        # State tracking
        self.current_state = "rally_end"
        self.state_duration = 0
        self.transitions = []

    def is_players_behind_service_line(self, p1_pos, p2_pos):
        """Check if both players are behind the service line"""
        if p1_pos is None or p2_pos is None:
            return False
        return p1_pos[1] >= self.service_line and p2_pos[1] >= self.service_line

    def is_players_on_different_sides(self, p1_pos, p2_pos):
        """Check if players are on different sides of the court"""
        if p1_pos is None or p2_pos is None:
            return False
        return (
            p1_pos[0] < self.court_center_x and p2_pos[0] > self.court_center_x
        ) or (p1_pos[0] > self.court_center_x and p2_pos[0] < self.court_center_x)

    def is_player_in_service_box(self, p1_pos, p2_pos):
        """Check if at least one player is in the service box area"""
        if p1_pos is None or p2_pos is None:
            return False

        def _is_in_box(pos):
            y_in_range = self.service_line <= pos[1] <= self.service_box_back
            x_in_left = self.left_box["x_min"] <= pos[0] <= self.left_box["x_max"]
            x_in_right = self.right_box["x_min"] <= pos[0] <= self.right_box["x_max"]
            return y_in_range and (x_in_left or x_in_right)

        return _is_in_box(p1_pos) or _is_in_box(p2_pos)

    def is_rally_start(self, p1_pos, p2_pos, combined_intensity):
        """Check if players are in rally start positions"""
        if p1_pos is None or p2_pos is None or combined_intensity is None:
            return False

        # Low intensity required
        if combined_intensity > self.start_intensity_threshold:
            return False

        # All positional conditions must be met
        return (
            self.is_players_on_different_sides(p1_pos, p2_pos)
            and self.is_players_behind_service_line(p1_pos, p2_pos)
            and self.is_player_in_service_box(p1_pos, p2_pos)
        )

    def is_rally_active(self, p1_pos, p2_pos, combined_intensity):
        """Check if conditions indicate active rally"""
        if p1_pos is None or p2_pos is None or combined_intensity is None:
            return False

        # Intensity should be high enough
        if combined_intensity < self.active_intensity_threshold:
            return False

        # Players should not be in start positions
        return not self.is_rally_start(p1_pos, p2_pos, 0.0)

    def is_rally_end(self, combined_intensity, p1_pos, p2_pos):
        """Check if conditions indicate rally end"""
        if p1_pos is None or p2_pos is None or combined_intensity is None:
            return False

        # Low intensity required
        if combined_intensity > self.end_intensity_threshold:
            return False

        # Players should be on different sides with low intensity
        return (
            self.is_players_on_different_sides(p1_pos, p2_pos)
            and combined_intensity < self.end_intensity_threshold
        )

    def update_state(
        self, combined_intensity, p1_pos, p2_pos, frame_count, avg_distance
    ):
        """Update rally state based on conditions and logical transitions"""
        new_state = self.current_state
        state_changed = False

        # Logical state transitions: rally_end -> rally_start -> rally_active -> rally_end
        if self.current_state == "rally_end":
            if self.is_rally_start(p1_pos, p2_pos, combined_intensity):
                new_state = "rally_start"
                state_changed = True

        elif self.current_state == "rally_start":
            if self.is_rally_active(p1_pos, p2_pos, combined_intensity):
                new_state = "rally_active"
                state_changed = True

        elif self.current_state == "rally_active":
            if self.is_rally_end(combined_intensity, p1_pos, p2_pos):
                new_state = "rally_end"
                state_changed = True

        # Handle state change
        if state_changed:
            self.transitions.append(
                {
                    "frame": frame_count,
                    "from_state": self.current_state,
                    "to_state": new_state,
                    "combined_intensity": combined_intensity,
                    "player_distance": self._calculate_distance(p1_pos, p2_pos),
                }
            )
            self.current_state = new_state
            self.state_duration = 0
        else:
            self.state_duration += 1

        return new_state, state_changed

    def _calculate_distance(self, p1_pos, p2_pos):
        """Calculate distance between two players"""
        if p1_pos is None or p2_pos is None:
            return float("inf")
        return np.sqrt((p1_pos[0] - p2_pos[0]) ** 2 + (p1_pos[1] - p2_pos[1]) ** 2)

    def get_summary(self):
        """Get summary of all rally transitions"""
        return {
            "final_state": self.current_state,
            "total_transitions": len(self.transitions),
            "transitions": self.transitions,
        }
