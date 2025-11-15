"""
Model Training Script

Trains the rally state detection model on annotated data.
"""

import os

from rally_state_detection.utilities.general import load_config
from rally_state_detection.model.ml_based_model import MLBasedModel


def main():
    """Train the model on annotation data."""
    # Get test directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(test_dir, "data")

    # Load package config
    config = load_config()

    # Initialize model
    print("Initializing ML model...")
    model = MLBasedModel(config)

    # Run training pipeline
    print(f"\nTraining model on data from: {data_dir}")
    accuracy = model.run_training_pipeline(
        data_path=data_dir, test_size=0.2, aggregated=True
    )

    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print("\nModel training complete!")


if __name__ == "__main__":
    main()
