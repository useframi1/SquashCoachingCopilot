import cv2


class VideoDisplay:
    """Handles video display with overlays for player tracking and rally states"""
    
    def __init__(self, config):
        self.config = config
        display_config = config.get("display", {})
        
        # Display settings
        self.show_bboxes = display_config.get("show_bounding_boxes", True)
        self.show_labels = display_config.get("show_player_labels", True)
        self.show_stats = display_config.get("show_statistics", True)
        self.show_rally_state = display_config.get("show_rally_state", True)
        
        # Colors for different elements
        self.colors = {
            "player1": (0, 255, 0),      # Green
            "player2": (255, 0, 0),      # Blue
            "rally_end": (0, 0, 255),    # Red
            "rally_start": (0, 255, 255), # Yellow
            "rally_active": (0, 255, 0),  # Green
            "text": (255, 255, 255),     # White
            "no_data": (0, 0, 255)       # Red
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = {
            "large": 1.0,
            "medium": 0.8,
            "small": 0.6
        }
        self.thickness = 2
    
    def draw_bounding_boxes(self, frame, pixel_coords):
        """Draw bounding boxes and player labels"""
        if not self.show_bboxes:
            return frame
        
        for player_id, bbox in pixel_coords.items():
            x1, y1, x2, y2 = map(int, bbox)
            color = self.colors[f"player{player_id}"]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
            
            if self.show_labels:
                # Draw player ID label
                label = f"Player {player_id}"
                label_size = cv2.getTextSize(label, self.font, self.font_scale["small"], self.thickness)[0]
                
                # Background rectangle for label
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1
                )
                
                # Label text
                cv2.putText(
                    frame, label, (x1, y1 - 5),
                    self.font, self.font_scale["small"], self.colors["text"], self.thickness
                )
        
        return frame
    
    def draw_frame_info(self, frame, frame_count):
        """Draw frame number"""
        cv2.putText(
            frame, f"Frame: {frame_count}", (10, 30),
            self.font, self.font_scale["large"], self.colors["text"], self.thickness
        )
        return frame
    
    def draw_rally_state(self, frame, rally_state, combined_intensity, avg_distance=None):
        """Draw rally state and intensity information"""
        if not self.show_rally_state:
            return frame
        
        # Rally state with color coding
        state_color = self.colors.get(rally_state, self.colors["text"])
        rally_text = f"Rally State: {rally_state.upper()}"
        
        cv2.putText(
            frame, rally_text, (frame.shape[1] - 400, 30),
            self.font, self.font_scale["medium"], state_color, self.thickness
        )
        
        # Combined intensity
        intensity_text = f"Combined Intensity: {combined_intensity:.3f}"
        cv2.putText(
            frame, intensity_text, (frame.shape[1] - 400, 65),
            self.font, self.font_scale["small"], self.colors["text"], self.thickness
        )
        
        # Average distance between players
        if avg_distance is not None:
            distance_text = f"Avg Distance: {avg_distance:.2f}m"
            cv2.putText(
                frame, distance_text, (frame.shape[1] - 400, 95),
                self.font, self.font_scale["small"], self.colors["text"], self.thickness
            )
        
        return frame
    
    def draw_player_stats(self, frame, player_stats):
        """Draw player statistics on frame"""
        if not self.show_stats:
            return frame
        
        y_offset = 70
        
        # Player 1 stats
        p1_stats = player_stats["player1"]
        if p1_stats["avg_position"] is not None:
            pos = p1_stats["avg_position"]
            intensity = p1_stats["intensity"]
            
            pos_text = f"P1 Avg: ({pos[0]:.2f}, {pos[1]:.2f})m"
            cv2.putText(
                frame, pos_text, (10, y_offset),
                self.font, self.font_scale["small"], self.colors["player1"], self.thickness
            )
            y_offset += 25
            
            intensity_text = f"P1 Intensity: {intensity:.3f}m/frame"
            cv2.putText(
                frame, intensity_text, (10, y_offset),
                self.font, self.font_scale["small"], self.colors["player1"], self.thickness
            )
            y_offset += 30
        else:
            cv2.putText(
                frame, "Player 1: No data", (10, y_offset),
                self.font, self.font_scale["small"], self.colors["no_data"], self.thickness
            )
            y_offset += 55
        
        # Player 2 stats
        p2_stats = player_stats["player2"]
        if p2_stats["avg_position"] is not None:
            pos = p2_stats["avg_position"]
            intensity = p2_stats["intensity"]
            
            pos_text = f"P2 Avg: ({pos[0]:.2f}, {pos[1]:.2f})m"
            cv2.putText(
                frame, pos_text, (10, y_offset),
                self.font, self.font_scale["small"], self.colors["player2"], self.thickness
            )
            y_offset += 25
            
            intensity_text = f"P2 Intensity: {intensity:.3f}m/frame"
            cv2.putText(
                frame, intensity_text, (10, y_offset),
                self.font, self.font_scale["small"], self.colors["player2"], self.thickness
            )
        else:
            cv2.putText(
                frame, "Player 2: No data", (10, y_offset),
                self.font, self.font_scale["small"], self.colors["no_data"], self.thickness
            )
        
        return frame
    
    def create_display_frame(self, frame, frame_count, pixel_coords, rally_state, 
                            combined_intensity, player_stats):
        """Create complete display frame with all overlays"""
        display_frame = frame.copy()
        
        # Draw all elements
        display_frame = self.draw_bounding_boxes(display_frame, pixel_coords)
        display_frame = self.draw_frame_info(display_frame, frame_count)
        display_frame = self.draw_rally_state(display_frame, rally_state, combined_intensity, 
                                            player_stats.get("avg_distance"))
        display_frame = self.draw_player_stats(display_frame, player_stats)
        
        return display_frame
    
    def show_frame(self, display_frame, window_name="Squash Player Tracking"):
        """Display frame and handle user input"""
        cv2.imshow(window_name, display_frame)
        return cv2.waitKey(1) & 0xFF
    
    def cleanup(self):
        """Clean up display resources"""
        cv2.destroyAllWindows()