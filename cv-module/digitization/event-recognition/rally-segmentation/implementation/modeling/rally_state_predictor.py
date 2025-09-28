"""
Clean ML Model Training Pipeline
Uses the unified feature engineering for consistent training.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
from typing import Dict, List

from feature_engineer import FeatureEngineer
from utilities.general import load_config, load_and_combine_data

warnings.filterwarnings("ignore")


class MLModelTrainer:
    """
    Clean ML model trainer using unified feature engineering.
    """

    def __init__(self):
        """Initialize the trainer."""
        self.config = load_config()
        self.model_config = self.config["rally_segmenter"]["ml_based"]
        self.model_type = self.model_config["model_type"]

        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.label_encoder = LabelEncoder()
        self.prev_state_encoder = LabelEncoder()
        self.feature_names = None
        self.is_fitted = False

        # Initialize model based on type
        if self.model_type == "xgboost":
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="mlogloss",
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        else:
            raise ValueError("model_type must be 'xgboost' or 'random_forest'")

    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare data for modeling using unified feature engineering.

        Args:
            df: Raw dataframe with base metrics

        Returns:
            Tuple of (X, y, groups, df_features)
        """
        print("Preparing data...")

        # Check data quality
        print(f"Original dataset shape: {df.shape}")
        print(f"Missing values per column:")
        missing_vals = df.isnull().sum()
        if missing_vals.sum() > 0:
            print(missing_vals[missing_vals > 0])
        else:
            print("No missing values found")

        # Drop any rows with missing values
        initial_rows = len(df)
        df_clean = df.dropna().reset_index(drop=True)
        print(f"Dropped {initial_rows - len(df_clean)} rows with missing values")

        # Check class distribution
        print("\nClass distribution:")
        print(df_clean["state"].value_counts())
        print("\nClass distribution (%):")
        print(df_clean["state"].value_counts(normalize=True) * 100)

        # Engineer features using unified feature engineer
        print("\nEngineering features...")
        df_features = self.feature_engineer.compute_features_batch(
            df_clean, group_col="video_name"
        )

        print("Encoding categorical variables...")

        # Fit encoders
        unique_states = ["start", "active", "end"]
        self.prev_state_encoder.fit(unique_states)
        y = self.label_encoder.fit_transform(df_features["state"])

        # Encode prev_state
        df_features["prev_state_encoded"] = self.prev_state_encoder.transform(
            df_features["prev_state"]
        )

        # Prepare feature matrix
        features_to_drop = [
            "frame_number",
            "window_size",
            "video_name",
            "state",
            "prev_state",
        ]

        # Only drop columns that exist
        existing_features_to_drop = [
            col for col in features_to_drop if col in df_features.columns
        ]
        X = df_features.drop(columns=existing_features_to_drop)

        # Store feature names
        self.feature_names = list(X.columns)

        # Groups for cross-validation
        groups = (
            df_features["video_name"] if "video_name" in df_features.columns else None
        )

        print(f"Final dataset shape: {X.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        if groups is not None:
            print(f"Number of videos: {groups.nunique()}")

        return X, y, groups, df_features

    def train_test_split_eval(self, X, y, groups, test_size=0.2):
        """
        Evaluate model using train-test split (group-aware).
        """
        print(f"\nEvaluating with train-test split (test_size={test_size})...")

        if groups is not None:
            # Group-aware split
            unique_groups = groups.unique()
            np.random.seed(42)
            np.random.shuffle(unique_groups)

            n_test_groups = max(1, int(len(unique_groups) * test_size))
            test_groups = unique_groups[:n_test_groups]
            train_groups = unique_groups[n_test_groups:]

            print(f"Train videos: {len(train_groups)}, Test videos: {len(test_groups)}")
            print(f"Test videos: {test_groups}")

            # Create train/test masks
            train_mask = groups.isin(train_groups)
            test_mask = groups.isin(test_groups)

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
        else:
            # Standard split if no groups
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f}")

        # Classification report
        target_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=target_names)
        print("\nClassification Report:")
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm,
            "y_test": y_test,
            "y_pred": y_pred,
            "target_names": target_names,
        }

    def get_feature_importance(self, top_n=15):
        """Get and display feature importance."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")

        importance = self.model.feature_importances_

        # Create importance dataframe
        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        print(f"\nTop {top_n} Feature Importance:")
        print(importance_df.head(top_n))

        return importance_df

    def plot_feature_importance(self, top_n=15, figsize=(10, 8)):
        """Plot feature importance."""
        importance_df = self.get_feature_importance()

        plt.figure(figsize=figsize)
        top_features = importance_df.head(top_n)

        sns.barplot(data=top_features, y="feature", x="importance")
        plt.title(f"Top {top_n} Feature Importance ({self.model_type.title()})")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, cm, target_names, figsize=(8, 6)):
        """Plot confusion matrix."""
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    def save_model(self, filepath):
        """Save the trained model and encoders."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")

        # Save model and encoders
        model_data = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "prev_state_encoder": self.prev_state_encoder,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
            "court_center_x": self.feature_engineer.court_center_x,
            "service_line_y": self.feature_engineer.service_line_y,
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def run(self):
        """Run the complete training pipeline."""
        print("=" * 60)
        print("ML MODEL TRAINING PIPELINE")
        print("=" * 60)

        # Load data
        print("\n1. LOADING DATA...")
        df = load_and_combine_data(self.config["annotations"]["output_path"])
        print(f"Loaded {len(df)} samples from {df['video_name'].nunique()} videos")

        # Prepare data (includes feature engineering)
        print("\n2. PREPARING DATA...")
        X, y, groups, df_features = self.prepare_data(df)

        # Train and evaluate
        print("\n3. TRAINING AND EVALUATION...")
        results = self.train_test_split_eval(X, y, groups, test_size=0.2)

        # Feature importance
        print("\n4. FEATURE IMPORTANCE...")
        self.plot_feature_importance(top_n=15)

        # Confusion matrix
        print("\n5. CONFUSION MATRIX...")
        self.plot_confusion_matrix(results["confusion_matrix"], results["target_names"])

        # Save model
        print("\n6. SAVING MODEL...")
        model_path = self.model_config["model_path"]
        self.save_model(model_path)

        print(f"\n{'='*60}")
        print("MODEL TRAINING COMPLETED!")
        print(f"Model saved to: {model_path}")
        print(f"Final accuracy: {results['accuracy']:.4f}")
        print(f"{'='*60}")

        return results


if __name__ == "__main__":
    trainer = MLModelTrainer()
    results = trainer.run()
