import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import plotly.graph_objects as go
from utilities.utils import load_config
import streamlit as st


class PositioningAnalysisPipeline:
    def __init__(
        self, court_dimensions, court_top_view_path, player_real_positions, fps
    ):
        self.config = load_config("configs/positioning_analysis_pipeline_config.json")
        self.court_width = court_dimensions[0]
        self.court_height = court_dimensions[1]
        self.court_top_view_path = court_top_view_path
        self.zones = {
            "top_left": tuple(self.config["court_zones"]["top_left"]),
            "top_right": tuple(self.config["court_zones"]["top_right"]),
            "bottom_left": tuple(self.config["court_zones"]["bottom_left"]),
            "bottom_right": tuple(self.config["court_zones"]["bottom_right"]),
        }
        self.player_real_positions = player_real_positions
        self.fps = fps

    def calculate_time_in_zones(self):
        time_in_zones = {
            pid: {zone: 0 for zone in self.zones} for pid in self.player_real_positions
        }

        for pid, positions in self.player_real_positions.items():
            for x, y in positions:
                for zone_name, (x1, y1, x2, y2) in self.zones.items():
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        time_in_zones[pid][zone_name] += 1

        for pid in time_in_zones:
            for zone in time_in_zones[pid]:
                time_in_zones[pid][zone] = round(time_in_zones[pid][zone] / self.fps, 2)

        return time_in_zones

    def calculate_distance_covered(self):
        distances = {}
        for pid, positions in self.player_real_positions.items():
            distance = 0
            for i in range(1, len(positions)):
                # Convert tuples to numpy arrays before subtraction
                pos_current = np.array(positions[i])
                pos_previous = np.array(positions[i - 1])

                # Calculate distance
                distance += np.linalg.norm(pos_current - pos_previous)

            distances[pid] = round(distance, 2)
        print(distances)
        return distances

    def calculate_average_speed(self):
        average_speeds = {}
        distances = self.calculate_distance_covered()
        time_in_zones = self.calculate_time_in_zones()

        for pid in distances:
            total_time = sum(time_in_zones[pid].values())
            if total_time > 0:
                average_speed = distances[pid] / total_time
                average_speeds[pid] = round(average_speed, 2)
            else:
                average_speeds[pid] = 0

        return average_speeds

    def plot_heatmaps(self):
        st.header("Player Positioning Heatmaps")
        court_img = cv2.imread(self.court_top_view_path)
        if court_img is None:
            raise FileNotFoundError(
                f"Could not load court image from {self.court_top_view_path}"
            )

        court_img = cv2.cvtColor(court_img, cv2.COLOR_BGR2RGB)
        img_height, img_width = court_img.shape[:2]

        heatmap_cols = st.columns(len(self.player_real_positions))

        for idx, (pid, positions) in enumerate(self.player_real_positions.items()):
            with heatmap_cols[idx]:
                st.subheader(f"Player {pid}")
                fig, ax = plt.subplots(figsize=(3, 3))
                positions = np.array(positions)
                x_scaled = (positions[:, 0] / self.court_width) * img_width
                y_scaled = (positions[:, 1] / self.court_height) * img_height

                heatmap = np.zeros((img_height, img_width), dtype=np.float32)
                for x, y in zip(x_scaled.astype(int), y_scaled.astype(int)):
                    if 0 <= x < img_width and 0 <= y < img_height:
                        heatmap[y, x] += 1

                heatmap = cv2.GaussianBlur(
                    heatmap,
                    tuple(self.config["heatmap"]["gaussian_blur"]["kernel_size"]),
                    self.config["heatmap"]["gaussian_blur"]["sigma_x"],
                )
                heatmap = np.sqrt(heatmap)
                heatmap = cv2.normalize(
                    heatmap,
                    None,
                    alpha=self.config["heatmap"]["normalization"]["alpha"],
                    beta=self.config["heatmap"]["normalization"]["beta"],
                    norm_type=cv2.NORM_MINMAX,
                )
                heatmap_color = cv2.applyColorMap(
                    heatmap.astype(np.uint8), cv2.COLORMAP_JET
                )

                overlay = cv2.addWeighted(
                    court_img,
                    self.config["heatmap"]["weights"]["alpha"],
                    heatmap_color,
                    self.config["heatmap"]["weights"]["beta"],
                    self.config["heatmap"]["weights"]["gamma"],
                )

                ax.imshow(overlay)
                ax.axis("off")
                st.pyplot(fig)

    def plot_time_in_zones(self):
        st.header("Time Spent in Court Zones (in seconds)")
        time_in_zones = self.calculate_time_in_zones()
        time_in_zones_cols = st.columns(len(time_in_zones))
        for idx, (pid, zones) in enumerate(time_in_zones.items()):
            with time_in_zones_cols[idx]:
                st.subheader(f"Player {pid}")
                st.bar_chart(data=zones, color=["#d32f2f"])

    def plot_distance_covered(self):
        st.header("Distance Covered")
        distances = self.calculate_distance_covered()

        col1, col2 = st.columns(2)

        with col1:
            for pid, distance in distances.items():
                if pid == 1:
                    st.markdown(
                        f"<div style='text-align: center;'>"
                        f"<span style='font-size: 30px; color: #d32f2f;'>Player {pid}</span><br>"
                        f"<span style='font-size: 70px; font-weight: bold; color: white;'>{distance} m</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        with col2:
            for pid, distance in distances.items():
                if pid == 2:
                    st.markdown(
                        f"<div style='text-align: center;'>"
                        f"<span style='font-size: 30px; color: #d32f2f;'>Player {pid}</span><br>"
                        f"<span style='font-size: 70px; font-weight: bold; color: white;'>{distance} m</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    def plot_average_speed(self):
        st.header("Average Speed")
        average_speeds = self.calculate_average_speed()

        col1, col2 = st.columns(2)

        with col1:
            for pid, speed in average_speeds.items():
                if pid == 1:
                    st.markdown(
                        f"<div style='text-align: center;'>"
                        f"<span style='font-size: 30px; color: #d32f2f;'>Player {pid}</span><br>"
                        f"<span style='font-size: 70px; font-weight: bold; color: white;'>{speed} m/s</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        with col2:
            for pid, speed in average_speeds.items():
                if pid == 2:
                    st.markdown(
                        f"<div style='text-align: center;'>"
                        f"<span style='font-size: 30px; color: #d32f2f;'>Player {pid}</span><br>"
                        f"<span style='font-size: 70px; font-weight: bold; color: white;'>{speed} m/s</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    def display_dashboard(self):
        col1, spacer, col2 = st.columns([3, 0.25, 3])
        with col1:
            self.plot_heatmaps()
        with col2:
            self.plot_time_in_zones()
            self.plot_distance_covered()
            self.plot_average_speed()
