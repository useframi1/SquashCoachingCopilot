import argparse


if __name__ == "__main__":
    # Import the actual dataset and model classes
    from squash_rally_dataset import SquashRallyDataset
    from squash_rally_detector import SquashRallyDetector
    from training_pipeline import TrainingPipeline
    from squash_rally_annotator import SquashAnnotator

    parser = argparse.ArgumentParser(description="Squash Rally Annotator")
    parser.add_argument(
        "--video_dir",
        type=str,
        default="./squash_videos",
        help="Directory containing video files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="annotations.json",
        help="Output JSON file for annotations",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=64,
        help="Sequence length (number of frames per sequence)",
    )

    args = parser.parse_args()

    # Configuration
    # CONFIG = {
    #     "sequence_length": 64,  # Number of frames in each sequence
    #     "image_size": 224,  # Resize frames to this size
    #     "batch_size": 8,
    #     "num_epochs": 50,
    #     "learning_rate": 1e-4,
    #     "feature_dim": 512,  # Dimension of CNN features
    #     "lstm_hidden_dim": 256,  # LSTM hidden layer size
    #     "dropout": 0.5,
    #     "data_path": "./videos/",
    #     "annotation_file": "./annotations.json",
    #     "model_save_path": "./models/",
    #     "early_stopping_patience": 10,
    # }

    annotator = SquashAnnotator(
        video_dir=args.video_dir,
        output_file=args.output,
        sequence_length=args.seq_length,
    )

    annotator.run()

    # pipeline = TrainingPipeline(CONFIG, SquashRallyDataset, SquashRallyDetector)

    # # Train the model
    # best_model_path = pipeline.train()
    # print(f"Training completed. Best model saved at: {best_model_path}")
