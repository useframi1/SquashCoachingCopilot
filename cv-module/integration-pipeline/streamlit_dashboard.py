import streamlit as st
from pipelines.digitization_pipeline import DigitizationPipeline
from pipelines.positioning_analysis_pipeline import PositioningAnalysisPipeline
from utilities.utils import load_config

st.set_page_config(layout="wide", page_title="Player Positioning Dashboard")
st.title("🏸 Player Positioning Dashboard")

config = load_config("configs/main_config.json")

with st.spinner("Running digitization pipeline..."):
    frame_placeholder = st.empty()
    digitizer = DigitizationPipeline(
        video_path=config["video_path"],
        court_dimensions=(
            config["court_dimensions"]["width"],
            config["court_dimensions"]["height"],
        ),
        court_top_view_path=config["court_top_view_path"],
        use_streamlit=True,
        streamlit_frame_placeholder=frame_placeholder,
    )
    player_real_positions, fps = digitizer.run()

with st.spinner("Running positioning analysis..."):
    analyzer = PositioningAnalysisPipeline(
        court_dimensions=(
            config["court_dimensions"]["width"],
            config["court_dimensions"]["height"],
        ),
        court_top_view_path=config["court_top_view_path"],
        player_real_positions=player_real_positions,
        fps=fps,
    )

    analyzer.display_dashboard()
