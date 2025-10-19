"""
Main entry point for the squash video analysis pipeline.

Simply imports the pipeline and runs it with the specified configuration.
"""

from pipeline import Pipeline
from config import PipelineConfig
import json


def main():
    """Run the complete squash video analysis pipeline."""

    # Configure the pipeline components
    config = PipelineConfig()

    # Initialize and run the pipeline
    pipeline = Pipeline(config=config)

    results = pipeline.run()

    # Save the results to a JSON file
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
