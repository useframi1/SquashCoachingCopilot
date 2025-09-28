"""
Model Trainer - Trains ML models for rally state prediction
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

from feature_engineer import FeatureEngineer
from config import CONFIG


class ModelTrainer:
    """Handles training of rally state prediction models."""

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False

        # Initialize model based on config
        if CONFIG["model_type"] == "xgboost":
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=CONFIG["random_seed"],
            )
        elif CONFIG["model_type"] == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=CONFIG["random_seed"]
            )
        else:
            raise ValueError(f"Unknown model type: {CONFIG['model_type']}")

    def load_data(self) -> pd.DataFrame:
        """Load and combine CSV files with base metrics."""
        print("Loading annotation data...")

        # Find all CSV files in data path
        csv_files = glob.glob(os.path.join(CONFIG["data_path"], "*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {CONFIG['data_path']}")

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

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features for training.

        Returns:
            Tuple of (X, y, video_names)
        """
        print("Engineering features...")

        # Engineer features grouped by video
        df_features = self.feature_engineer.engineer_features(
            df, group_col="video_name"
        )

        # Get feature names
        self.feature_names = self.feature_engineer.get_feature_names()

        # Prepare feature matrix and labels
        X = df_features[self.feature_names]
        y = self.label_encoder.fit_transform(df_features["state"])
        video_names = df_features["video_name"]

        print(f"Features prepared: {X.shape}")
        print(f"Classes: {list(self.label_encoder.classes_)}")

        return X, y, video_names

    def train_test_split_by_video(self, X, y, video_names):
        """Split data by videos to avoid data leakage."""
        unique_videos = video_names.unique()

        # Split videos into train/test
        train_videos, test_videos = train_test_split(
            unique_videos,
            test_size=CONFIG["test_size"],
            random_state=CONFIG["random_seed"],
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
        print(f"Training {CONFIG['model_type']} model...")

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

    def save_model(self):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        model_data = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "model_type": CONFIG["model_type"],
        }

        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(CONFIG["model_path"]), exist_ok=True)

        joblib.dump(model_data, CONFIG["model_path"])
        print(f"Model saved to {CONFIG['model_path']}")

    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        print("=" * 60)
        print("RALLY STATE PREDICTION - TRAINING PIPELINE")
        print("=" * 60)

        # Load data
        df = self.load_data()

        # Prepare features
        X, y, video_names = self.prepare_features(df)

        # Train/test split by video
        X_train, X_test, y_train, y_test = self.train_test_split_by_video(
            X, y, video_names
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


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_training_pipeline()
