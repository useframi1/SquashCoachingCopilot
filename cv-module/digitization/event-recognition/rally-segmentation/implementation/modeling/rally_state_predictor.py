"""
Rally State Predictor
A machine learning model to predict squash rally states (start, active, end) using XGBoost.
Includes data preprocessing, training, evaluation, and prediction capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings

warnings.filterwarnings("ignore")


class RallyStatePredictor:
    """
    A machine learning model for predicting rally states in squash videos.
    """

    def __init__(self, model_type="xgboost"):
        """
        Initialize the predictor.

        Args:
            model_type: 'xgboost' or 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.prev_state_encoder = LabelEncoder()
        self.feature_names = None
        self.is_fitted = False

        # Initialize model based on type
        if model_type == "xgboost":
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="mlogloss",
            )
        elif model_type == "random_forest":
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

    def prepare_data(self, df):
        """
        Prepare data for modeling by handling missing values and encoding.

        Args:
            df: Raw dataframe with all features

        Returns:
            Tuple of (X, y, groups, df_clean)
        """
        print("Preparing data...")

        # Make a copy to avoid modifying original
        df_clean = df.copy()

        # Check data quality
        print(f"Original dataset shape: {df_clean.shape}")
        print(f"Missing values per column:")
        print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])

        # Drop any remaining rows with missing values
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna().reset_index(drop=True)
        print(f"Dropped {initial_rows - len(df_clean)} rows with missing values")

        # Check class distribution
        print("\nClass distribution:")
        print(df_clean["state"].value_counts())
        print("\nClass distribution (%):")
        print(df_clean["state"].value_counts(normalize=True) * 100)

        # Encode categorical variables
        print("\nEncoding categorical variables...")

        # Encode prev_state
        df_clean["prev_state_encoded"] = self.prev_state_encoder.fit_transform(
            df_clean["prev_state"]
        )

        # Encode target variable
        y = self.label_encoder.fit_transform(df_clean["state"])

        # Prepare feature matrix
        features_to_drop = [
            "frame_number",
            "window_size",
            "video_name",
            "state",
            "prev_state",
        ]
        X = df_clean.drop(columns=features_to_drop)

        # Store feature names
        self.feature_names = list(X.columns)

        # Groups for cross-validation (video names)
        groups = df_clean["video_name"]

        print(f"Final dataset shape: {X.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Number of videos: {groups.nunique()}")

        return X, y, groups, df_clean

    def train_with_cv(self, X, y, groups, cv_folds=5):
        """
        Train model using GroupKFold cross-validation.

        Args:
            X: Feature matrix
            y: Target labels
            groups: Group labels (video names)
            cv_folds: Number of CV folds

        Returns:
            Cross-validation scores
        """
        print(f"\nTraining {self.model_type} with {cv_folds}-fold GroupKFold CV...")

        # Use GroupKFold to prevent data leakage between videos
        gkf = GroupKFold(n_splits=cv_folds)

        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, X, y, cv=gkf, groups=groups, scoring="accuracy", n_jobs=-1
        )

        print(f"Cross-validation scores: {cv_scores}")
        print(
            f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )

        # Train final model on all data
        print("Training final model on all data...")
        self.model.fit(X, y)
        self.is_fitted = True

        return cv_scores

    def train_test_split_eval(self, X, y, groups, test_size=0.2):
        """
        Evaluate model using train-test split (group-aware).

        Args:
            X: Feature matrix
            y: Target labels
            groups: Group labels (video names)
            test_size: Fraction for test set

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating with train-test split (test_size={test_size})...")

        # Get unique groups and split them
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
        """
        Get and display feature importance.

        Args:
            top_n: Number of top features to display

        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first")

        if self.model_type == "xgboost":
            importance = self.model.feature_importances_
        else:  # random_forest
            importance = self.model.feature_importances_

        # Create importance dataframe
        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        print(f"\nTop {top_n} Feature Importance:")
        print(importance_df.head(top_n))

        return importance_df

    def plot_feature_importance(self, top_n=15, figsize=(10, 8)):
        """
        Plot feature importance.

        Args:
            top_n: Number of top features to plot
            figsize: Figure size
        """
        importance_df = self.get_feature_importance()

        plt.figure(figsize=figsize)
        top_features = importance_df.head(top_n)

        sns.barplot(data=top_features, y="feature", x="importance")
        plt.title(f"Top {top_n} Feature Importance ({self.model_type.title()})")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, cm, target_names, figsize=(8, 6)):
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix
            target_names: Class names
            figsize: Figure size
        """
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

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first")

        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions)

    def predict_proba(self, X):
        """
        Get prediction probabilities.

        Args:
            X: Feature matrix

        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first")

        return self.model.predict_proba(X)

    def save_model(self, filepath):
        """
        Save the trained model and encoders.

        Args:
            filepath: Path to save the model (without extension)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first")

        # Save model and encoders
        model_data = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "prev_state_encoder": self.prev_state_encoder,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
        }

        joblib.dump(model_data, f"{filepath}.pkl")
        print(f"Model saved to {filepath}.pkl")

    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded RallyStatePredictor instance
        """
        model_data = joblib.load(filepath)

        # Create new instance
        predictor = cls(model_type=model_data["model_type"])
        predictor.model = model_data["model"]
        predictor.label_encoder = model_data["label_encoder"]
        predictor.prev_state_encoder = model_data["prev_state_encoder"]
        predictor.feature_names = model_data["feature_names"]
        predictor.is_fitted = True

        return predictor


def main():
    """
    Main function to demonstrate model training and evaluation.
    """
    # Load data
    print("Loading data...")
    df = pd.read_csv("data/annotations/combined_annotations.csv")

    # Initialize predictor
    predictor = RallyStatePredictor(model_type="xgboost")

    # Prepare data
    X, y, groups, df_clean = predictor.prepare_data(df)

    # Method 1: Cross-validation
    cv_scores = predictor.train_with_cv(X, y, groups, cv_folds=5)

    # Method 2: Train-test split evaluation
    # predictor = RallyStatePredictor(model_type='xgboost')  # Fresh instance
    # results = predictor.train_test_split_eval(X, y, groups, test_size=0.2)

    # Feature importance
    predictor.plot_feature_importance(top_n=15)

    # Save model
    predictor.save_model("models/rally_state_model")

    print("\nModel training completed!")

    # Example: Load and use saved model
    # loaded_predictor = RallyStatePredictor.load_model('models/rally_state_model.pkl')
    # predictions = loaded_predictor.predict(X[:10])
    # print(f"Sample predictions: {predictions}")


if __name__ == "__main__":
    main()
