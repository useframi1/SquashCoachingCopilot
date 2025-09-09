from pipelines.digitization_pipeline import DigitizationPipeline
from pipelines.positioning_analysis_pipeline import PositioningAnalysisPipeline
from utilities.utils import load_config

if __name__ == "__main__":
    try:
        config = load_config("configs/main_config.json")

        player_positioning_pipeline = DigitizationPipeline(
            video_path=config["video_path"],
            court_dimensions=(
                config["court_dimensions"]["width"],
                config["court_dimensions"]["height"],
            ),
            court_top_view_path=config["court_top_view_path"],
        )
        player_real_positions, fps = player_positioning_pipeline.run()

        # positioning_analysis_pipeline = PositioningAnalysisPipeline(
        #     court_dimensions=(
        #         config["court_dimensions"]["width"],
        #         config["court_dimensions"]["height"],
        #     ),
        #     court_top_view_path=config["court_top_view_path"],
        #     player_real_positions=player_real_positions,
        #     fps=fps,
        # )
        # positioning_analysis_pipeline.analyze()
    except Exception as e:
        print(f"An error occurred: {e}")
