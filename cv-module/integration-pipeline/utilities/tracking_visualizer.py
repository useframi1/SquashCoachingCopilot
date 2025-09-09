import cv2


class TrackingVisualizer:
    def __init__(self, court_dimensions, court_top_view_path, config):
        self.config = config
        self.court_width, self.court_height = court_dimensions
        self.court_overlay = None
        self.overlay_size = (
            config["mini_court_overlay"]["width"],
            config["mini_court_overlay"]["height"],
        )
        if court_top_view_path:
            court_img = cv2.imread(court_top_view_path)
            court_img = cv2.resize(court_img, self.overlay_size)
            self.court_overlay = court_img

    def visualize_tracking(
        self, frame, player_pixel_coordinates, player_real_coordinates
    ):
        frame_bboxes = self.draw_bboxes(frame, player_pixel_coordinates)
        frame_court_overlay = self.draw_court_overlay(
            frame_bboxes, player_real_coordinates
        )

        return frame_court_overlay

    def add_text(self, frame, text, position, font_scale, color, thickness):
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
        )

    def draw_bboxes(self, frame, player_pixel_coordinates):
        bboxes_config = self.config["bboxes"]
        for pid, (x1, y1, x2, y2) in player_pixel_coordinates.items():
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                tuple(bboxes_config["color"]),
                bboxes_config["thickness"],
            )
            self.add_text(
                frame,
                f"Player {pid}",
                (int(x1), int(y1) - 10),
                bboxes_config["font_scale"],
                tuple(bboxes_config["text_color"]),
                bboxes_config["text_thickness"],
            )

        return frame

    def draw_court_overlay(self, frame, player_real_coordinates):
        mini_court_config = self.config["mini_court_overlay"]
        if self.court_overlay is not None:
            overlay_copy = self.court_overlay.copy()

            for pid, positions in player_real_coordinates.items():
                if positions:
                    real_x, real_y = positions[-1]

                    scaled_x = int((real_x / self.court_width) * self.overlay_size[0])
                    scaled_y = int((real_y / self.court_height) * self.overlay_size[1])

                    cv2.circle(
                        overlay_copy,
                        (scaled_x, scaled_y),
                        mini_court_config["circle_radius"],
                        tuple(mini_court_config["circle_color"]),
                        -1,
                    )
                    self.add_text(
                        overlay_copy,
                        f"{pid}",
                        (scaled_x + 5, scaled_y - 5),
                        mini_court_config["font_scale"],
                        tuple(mini_court_config["text_color"]),
                        mini_court_config["text_thickness"],
                    )

            h, w = frame.shape[:2]
            x_offset = w - self.overlay_size[0] - 10
            y_offset = h - self.overlay_size[1] - 10
            frame[
                y_offset : y_offset + self.overlay_size[1],
                x_offset : x_offset + self.overlay_size[0],
            ] = overlay_copy

        return frame
