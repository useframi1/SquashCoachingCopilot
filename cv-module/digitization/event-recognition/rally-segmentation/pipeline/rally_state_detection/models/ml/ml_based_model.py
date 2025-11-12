"""
ML-Based Model - Machine learning model for rally state prediction
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import os
import glob
from collections import deque

from rally_state_detection.utilities.metrics_aggregator import MetricsAggregator
from rally_state_detection.utilities.general import get_package_dir


class MLBasedModel:
    """Handles training and inference of ML models for rally state prediction."""

    def __init__(self, config: dict):
        self.config = config
        self.metrics_aggregator = MetricsAggregator(
            window_size=self.config["window_size"],
            config=self.config
        )
        self.metrics_history = deque(
            maxlen=self.config["feature_engineering"]["lookback_frames"] + 1
        )
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        self.model_type = self.config["models"]["ml_based"]["model_type"]

        # Try to load trained model
        try:
            self.load_trained_model()
        except FileNotFoundError:
            # Model not found, will need to train
            self._initialize_model()

    def _initialize_model(self):
        """Initialize a new model based on config."""
        if self.model_type == "xgboost":
            self.model = XGBClassifier(
                n_estimators=self.config["models"]["ml_based"]["n_estimators"],
                max_depth=self.config["models"]["ml_based"]["max_depth"],
                learning_rate=self.config["models"]["ml_based"]["learning_rate"],
                random_state=42,
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.config["models"]["ml_based"]["n_estimators"],
                max_depth=self.config["models"]["ml_based"]["max_depth"],
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def load_trained_model(self):
        """Load a pre-trained model for inference."""
        model_path = os.path.join(
            get_package_dir(), self.config["models"]["ml_based"]["model_path"]
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found at {model_path}")

        model_data = joblib.load(model_path)
        self.model = model_data["model"]
        self.label_encoder = model_data["label_encoder"]
        self.feature_names = model_data["feature_names"]
        self.model_type = model_data["model_type"]
        self.is_trained = True

    def set_state(self, state: str):
        """Reset any internal state of the model."""
        pass

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict rally states for batch of data.

        Args:
            df: DataFrame with engineered features

        Returns:
            DataFrame with 'predicted_state' column added
        """
        if not self.is_trained:
            raise ValueError("Model must be trained or loaded first")

        df = df.copy()

        # Prepare feature matrix
        X = df[self.feature_names].values

        # Make predictions
        y_pred = self.model.predict(X)
        predictions = self.label_encoder.inverse_transform(y_pred)

        # Get prediction probabilities
        y_proba = self.model.predict_proba(X)

        # Extract confidence (probability of the predicted class)
        confidence = np.max(y_proba, axis=1)

        df["predicted_state"] = predictions
        df["prediction_confidence"] = confidence

        return df

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and combine CSV files with base metrics."""
        print("Loading annotation data...")

        # Find all CSV files in data path
        csv_files = glob.glob(os.path.join(data_path, "**/*.csv"), recursive=True)

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")

        # Load and combine all CSV files
        dataframes = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)

            # Add video name from filename if not present
            if "video_name" not in df.columns:
                video_name = os.path.splitext(os.path.basename(csv_file))[0]
                df["video_name"] = video_name

            dataframes.append(df)
            print(f"  Loaded {len(df)} samples from {csv_file}")

        combined_df = pd.concat(dataframes, ignore_index=True)
        print(
            f"Combined dataset: {len(combined_df)} samples from {len(csv_files)} videos"
        )

        return combined_df

    def prepare_features(self, df: pd.DataFrame, aggregated: bool = False) -> tuple:
        """
        Prepare features for training.

        Args:
            df: DataFrame with metrics. Can be either:
                - Frame-by-frame metrics (aggregated=False): player_distance, player1_x, player1_y, player2_x, player2_y
                - Aggregated metrics (aggregated=True): mean_distance, median_player1_x, median_player1_y, etc.
            aggregated: Whether the input data is already aggregated by window size

        Returns:
            Tuple of (X, y, video_names)
        """
        print("Engineering features...")

        # Check if data needs aggregation
        if not aggregated:
            # Data is frame-by-frame, need to aggregate first
            print("Aggregating frame-by-frame metrics...")

            # Process and engineer features grouped by video
            video_groups = []
            for video_name, group_df in df.groupby("video_name"):
                group_df = group_df.sort_values("frame_number").reset_index(drop=True)

                # Convert to list of dicts for process_and_engineer
                metrics_list = group_df.to_dict('records')

                # Use process_and_engineer to aggregate and engineer features
                group_features = self.metrics_aggregator.process_and_engineer(
                    metrics_list, aggregated=False
                )

                # Add video_name and state back (they were preserved in the aggregation)
                if 'video_name' not in group_features.columns:
                    group_features['video_name'] = video_name

                video_groups.append(group_features)
        else:
            # Data is already aggregated, just engineer features
            print("Data already aggregated, engineering features...")

            video_groups = []
            for video_name, group_df in df.groupby("video_name"):
                group_df = group_df.sort_values("frame_number").reset_index(drop=True)
                group_features = self.metrics_aggregator.engineer_features(group_df)
                video_groups.append(group_features)

        df_features = pd.concat(video_groups, ignore_index=True)

        # Get feature names
        self.feature_names = self.metrics_aggregator.get_feature_names()

        # Prepare feature matrix and labels
        X = df_features[self.feature_names]
        y = self.label_encoder.fit_transform(df_features["state"])
        video_names = df_features["video_name"]

        print(f"Features prepared: {X.shape}")
        print(f"Classes: {list(self.label_encoder.classes_)}")

        return X, y, video_names

    def train_test_split_by_video(self, X, y, video_names, test_size=0.2):
        """Split data by videos to avoid data leakage."""
        unique_videos = video_names.unique()

        # Split videos into train/test
        train_videos, test_videos = train_test_split(
            unique_videos,
            test_size=test_size,
            random_state=42,
        )

        print(f"Train videos: {list(train_videos)}")
        print(f"Test videos: {list(test_videos)}")

        # Create train/test masks
        train_mask = video_names.isin(train_videos)
        test_mask = video_names.isin(test_videos)

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """Train the model."""
        print(f"Training {self.model_type} model...")

        self.model.fit(X_train, y_train)
        self.is_trained = True

        print("Training completed!")

    def evaluate(self, X_test, y_test):
        """Evaluate the model."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        print("Evaluating model...")

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")

        # Classification report
        target_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=target_names)
        print("\nClassification Report:")
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print("(Rows: True, Columns: Predicted)")
        print(f"{'':>10} {'Start':<10} {'Active':<10} {'End':<10}")
        for i, state in enumerate(["Start", "Active", "End"]):
            print(f"{state:>10} {cm[i][0]:<10} {cm[i][1]:<10} {cm[i][2]:<10}")

        return accuracy, report, cm

    def save_model(self, output_path: str = None):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        if output_path is None:
            output_path = os.path.join(
                get_package_dir(), self.config["models"]["ml_based"]["model_path"]
            )

        model_data = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
        }

        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        joblib.dump(model_data, output_path)
        print(f"Model saved to {output_path}")

    def run_training_pipeline(self, data_path: str, test_size: float = 0.2, aggregated: bool = False):
        """
        Run the complete training pipeline.

        Args:
            data_path: Path to directory containing CSV files with metrics
            test_size: Fraction of videos to use for testing (default: 0.2)
            aggregated: Whether the loaded data is already aggregated (default: False, assumes frame-by-frame data)
        """
        print("=" * 60)
        print("RALLY STATE PREDICTION - TRAINING PIPELINE")
        print("=" * 60)

        # Load data
        df = self.load_data(data_path)

        # Prepare features
        X, y, video_names = self.prepare_features(df, aggregated=aggregated)

        # Train/test split by video
        X_train, X_test, y_train, y_test = self.train_test_split_by_video(
            X, y, video_names, test_size=test_size
        )

        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Train model
        self.train(X_train, y_train)

        # Evaluate
        accuracy, report, cm = self.evaluate(X_test, y_test)

        # Save model
        self.save_model()

        print("=" * 60)
        print("TRAINING COMPLETED!")
        print(f"Final Test Accuracy: {accuracy:.4f}")
        print("=" * 60)

        return accuracy
