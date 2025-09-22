#!/usr/bin/env python3
"""
Rally Segmentation Analysis and Modeling Pipeline

This script provides comprehensive analysis of manually annotated rally data
and builds models for automatic rally state classification.

Features:
- Exploratory Data Analysis with visualizations
- Feature engineering and statistical analysis
- Multiple modeling approaches (statistical thresholds and ML)
- Model evaluation and comparison
- Automated rally segmentation pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
import warnings

warnings.filterwarnings("ignore")


class AnalysisPipeline:
    def __init__(self, annotation_files):
        """
        Initialize analyzer with annotation CSV files

        Args:
            annotation_files: List of paths to annotation CSV files
        """
        self.annotation_files = annotation_files
        self.data = None
        self.features = None
        self.models = {}
        self.thresholds = {}

    def load_and_combine_data(self):
        """Load and combine all annotation files"""
        dfs = []
        for file in self.annotation_files:
            df = pd.read_csv(file)
            dfs.append(df)

        self.data = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.data)} annotations from {len(dfs)} files")
        print(f"States distribution:\n{self.data['state'].value_counts()}")

        return self.data

    def exploratory_data_analysis(self, save_plots=True):
        """Perform comprehensive EDA"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_and_combine_data() first.")

        # Set up the plotting style
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle("Rally Segmentation - Exploratory Data Analysis", fontsize=16)

        # 1. Distance distributions by state
        axes[0, 0].set_title("Player Distance by Rally State")
        for state in self.data["state"].unique():
            state_data = self.data[self.data["state"] == state][
                "mean_distance"
            ].dropna()
            if len(state_data) > 0:
                axes[0, 0].hist(state_data, alpha=0.7, label=state, bins=20)
        axes[0, 0].set_xlabel("Mean Distance")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].legend()

        # 2. Combined intensity distributions
        axes[0, 1].set_title("Combined Intensity by Rally State")
        for state in self.data["state"].unique():
            state_data = self.data[self.data["state"] == state][
                "mean_combined_intensity"
            ].dropna()
            if len(state_data) > 0:
                axes[0, 1].hist(state_data, alpha=0.7, label=state, bins=20)
        axes[0, 1].set_xlabel("Mean Combined Intensity")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].legend()

        # 3. Box plots for distance
        axes[0, 2].set_title("Distance Distribution by State")
        self.data.boxplot(column="mean_distance", by="state", ax=axes[0, 2])
        axes[0, 2].set_xlabel("Rally State")
        axes[0, 2].set_ylabel("Mean Distance")

        # 4. Box plots for intensity
        axes[1, 0].set_title("Intensity Distribution by State")
        self.data.boxplot(column="mean_combined_intensity", by="state", ax=axes[1, 0])
        axes[1, 0].set_xlabel("Rally State")
        axes[1, 0].set_ylabel("Mean Combined Intensity")

        # 5. Player positions scatter plot
        axes[1, 1].set_title("Player Positions by State")
        for state in self.data["state"].unique():
            state_data = self.data[self.data["state"] == state]
            valid_data = state_data.dropna(
                subset=["median_player1_x", "median_player1_y"]
            )
            if len(valid_data) > 0:
                axes[1, 1].scatter(
                    valid_data["median_player1_x"],
                    valid_data["median_player1_y"],
                    alpha=0.6,
                    label=f"Player 1 - {state}",
                    s=30,
                )
        axes[1, 1].set_xlabel("Court X Position")
        axes[1, 1].set_ylabel("Court Y Position")
        axes[1, 1].legend()

        # 6. Correlation heatmap
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[numeric_cols].corr()
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", center=0, ax=axes[1, 2], fmt=".2f"
        )
        axes[1, 2].set_title("Feature Correlation Matrix")

        # 7. Distance vs Intensity scatter
        axes[2, 0].set_title("Distance vs Intensity by State")
        for state in self.data["state"].unique():
            state_data = self.data[self.data["state"] == state]
            valid_data = state_data.dropna(
                subset=["mean_distance", "mean_combined_intensity"]
            )
            if len(valid_data) > 0:
                axes[2, 0].scatter(
                    valid_data["mean_distance"],
                    valid_data["mean_combined_intensity"],
                    alpha=0.6,
                    label=state,
                    s=30,
                )
        axes[2, 0].set_xlabel("Mean Distance")
        axes[2, 0].set_ylabel("Mean Combined Intensity")
        axes[2, 0].legend()

        # 8. Time series example (if frame numbers available)
        if "frame_number" in self.data.columns:
            # Show one video's progression
            sample_video = self.data["video_name"].iloc[0]
            video_data = self.data[self.data["video_name"] == sample_video].sort_values(
                "frame_number"
            )

            axes[2, 1].set_title(f"Rally Progression - {sample_video}")
            axes[2, 1].plot(
                video_data["frame_number"],
                video_data["mean_distance"],
                label="Distance",
                alpha=0.7,
            )
            ax2 = axes[2, 1].twinx()
            ax2.plot(
                video_data["frame_number"],
                video_data["mean_combined_intensity"],
                color="orange",
                label="Intensity",
                alpha=0.7,
            )

            # Color code by state
            for state in video_data["state"].unique():
                state_data = video_data[video_data["state"] == state]
                axes[2, 1].scatter(
                    state_data["frame_number"],
                    state_data["mean_distance"],
                    c={"start": "green", "active": "red", "end": "blue"}.get(
                        state, "gray"
                    ),
                    s=50,
                    alpha=0.8,
                    label=f"{state} annotations",
                )

            axes[2, 1].set_xlabel("Frame Number")
            axes[2, 1].set_ylabel("Distance")
            ax2.set_ylabel("Intensity")
            axes[2, 1].legend(loc="upper left")

        # 9. Summary statistics table
        axes[2, 2].axis("off")
        summary_stats = (
            self.data.groupby("state")[["mean_distance", "mean_combined_intensity"]]
            .agg(["count", "mean", "std", "median"])
            .round(4)
        )

        table_text = "Summary Statistics by State:\n\n"
        for state in summary_stats.index:
            table_text += f"{state.upper()}:\n"
            table_text += f"  Distance: μ={summary_stats.loc[state, ('mean_distance', 'mean')]:.3f}, "
            table_text += (
                f"σ={summary_stats.loc[state, ('mean_distance', 'std')]:.3f}\n"
            )
            table_text += f"  Intensity: μ={summary_stats.loc[state, ('mean_combined_intensity', 'mean')]:.6f}, "
            table_text += f"σ={summary_stats.loc[state, ('mean_combined_intensity', 'std')]:.6f}\n\n"

        axes[2, 2].text(
            0.1,
            0.9,
            table_text,
            transform=axes[2, 2].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()
        if save_plots:
            plt.savefig("rally_segmentation_eda.png", dpi=300, bbox_inches="tight")
        plt.show()

        return summary_stats

    def statistical_threshold_analysis(self):
        """Analyze statistical thresholds for rule-based classification"""
        print("=== Statistical Threshold Analysis ===")

        results = {}
        for feature in ["mean_distance", "mean_combined_intensity"]:
            print(f"\nAnalyzing {feature}:")

            feature_stats = {}
            for state in ["start", "active", "end"]:
                data = self.data[self.data["state"] == state][feature].dropna()
                if len(data) > 0:
                    feature_stats[state] = {
                        "mean": data.mean(),
                        "std": data.std(),
                        "median": data.median(),
                        "q25": data.quantile(0.25),
                        "q75": data.quantile(0.75),
                        "count": len(data),
                    }
                    print(
                        f"  {state}: μ={data.mean():.4f}, σ={data.std():.4f}, "
                        f"median={data.median():.4f}"
                    )

            # Statistical tests
            groups = [
                self.data[self.data["state"] == state][feature].dropna()
                for state in ["start", "active", "end"]
            ]
            groups = [g for g in groups if len(g) > 0]  # Remove empty groups

            if len(groups) >= 2:
                # ANOVA test
                f_stat, p_value = stats.f_oneway(*groups)
                print(f"  ANOVA: F={f_stat:.4f}, p={p_value:.4f}")

                # Pairwise t-tests
                from itertools import combinations

                states = [
                    s
                    for s in ["start", "active", "end"]
                    if len(self.data[self.data["state"] == s][feature].dropna()) > 0
                ]

                for state1, state2 in combinations(states, 2):
                    data1 = self.data[self.data["state"] == state1][feature].dropna()
                    data2 = self.data[self.data["state"] == state2][feature].dropna()
                    if len(data1) > 0 and len(data2) > 0:
                        t_stat, p_val = stats.ttest_ind(data1, data2)
                        print(f"  {state1} vs {state2}: t={t_stat:.4f}, p={p_val:.4f}")

            results[feature] = feature_stats

        self.thresholds = results
        return results

    def engineer_features(self):
        """Create additional features for modeling"""
        df = self.data.copy()

        # Basic features
        features = [
            "mean_distance",
            "median_distance",
            "mean_combined_intensity",
            "median_combined_intensity",
            "mean_player1_intensity",
            "mean_player2_intensity",
        ]

        # Position-based features
        position_features = []
        if all(
            col in df.columns
            for col in [
                "median_player1_x",
                "median_player1_y",
                "median_player2_x",
                "median_player2_y",
            ]
        ):
            # Distance from court center (assuming center is at 0,0)
            df["player1_distance_from_center"] = np.sqrt(
                df["median_player1_x"] ** 2 + df["median_player1_y"] ** 2
            )
            df["player2_distance_from_center"] = np.sqrt(
                df["median_player2_x"] ** 2 + df["median_player2_y"] ** 2
            )

            # Court positioning ratios
            df["position_ratio_x"] = df["median_player1_x"] / (
                df["median_player2_x"] + 1e-8
            )
            df["position_ratio_y"] = df["median_player1_y"] / (
                df["median_player2_y"] + 1e-8
            )

            position_features = [
                "player1_distance_from_center",
                "player2_distance_from_center",
                "position_ratio_x",
                "position_ratio_y",
            ]
            features.extend(position_features)

        # Interaction features
        if "mean_distance" in df.columns and "mean_combined_intensity" in df.columns:
            df["distance_intensity_ratio"] = df["mean_distance"] / (
                df["mean_combined_intensity"] + 1e-8
            )
            df["distance_intensity_product"] = (
                df["mean_distance"] * df["mean_combined_intensity"]
            )
            features.extend(["distance_intensity_ratio", "distance_intensity_product"])

        # Filter to available features
        available_features = [f for f in features if f in df.columns]

        # Remove rows with too many missing values
        df_features = df[available_features + ["state"]].copy()
        df_features = df_features.dropna(
            subset=available_features, thresh=len(available_features) // 2
        )

        self.features = df_features
        print(
            f"Engineered {len(available_features)} features for {len(df_features)} samples"
        )
        print(f"Features: {available_features}")

        return df_features

    def build_models(self):
        """Build and evaluate multiple models"""
        if self.features is None:
            self.engineer_features()

        # Prepare data
        X = self.features.drop("state", axis=1)
        y = self.features["state"]

        # Handle missing values
        X = X.fillna(X.median())

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Define models
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Logistic Regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(random_state=42)),
                ]
            ),
            "SVM": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svm", SVC(random_state=42, probability=True)),
                ]
            ),
        }

        # Train and evaluate models
        results = {}
        for name, model in models.items():
            print(f"\n=== {name} ===")

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring="accuracy"
            )
            print(
                f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
            )

            # Train on full training set
            model.fit(X_train, y_train)

            # Test set evaluation
            y_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            print(f"Test Accuracy: {test_accuracy:.4f}")

            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            # Feature importance (if available)
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            elif hasattr(
                model.named_steps.get("lr", model.named_steps.get("svm")), "coef_"
            ):
                importance = np.abs(
                    model.named_steps.get("lr", model.named_steps.get("svm")).coef_[0]
                )
            else:
                importance = None

            if importance is not None:
                feature_importance = pd.DataFrame(
                    {"feature": X.columns, "importance": importance}
                ).sort_values("importance", ascending=False)
                print("\nTop 5 Most Important Features:")
                print(feature_importance.head())

            results[name] = {
                "model": model,
                "cv_accuracy": cv_scores.mean(),
                "test_accuracy": test_accuracy,
                "predictions": y_pred,
                "feature_importance": (
                    feature_importance if importance is not None else None
                ),
            }

        self.models = results
        return results

    def create_rule_based_classifier(self):
        """Create simple rule-based classifier using statistical thresholds"""
        if not self.thresholds:
            self.statistical_threshold_analysis()

        def classify_rally_state(distance, intensity):
            """
            Simple rule-based classification
            Customize these rules based on your threshold analysis
            """
            # These are example thresholds - adjust based on your data analysis

            # High intensity usually indicates active rally
            if intensity > 0.05:  # Adjust threshold based on your data
                return "active"

            # Low intensity with medium distance might be start/end
            elif intensity < 0.01:
                if distance > 5.0:  # Adjust based on your court size
                    return "start"  # Players far apart
                else:
                    return "end"  # Players close together

            # Default to active for medium intensity
            else:
                return "active"

        return classify_rally_state

    def evaluate_on_new_data(self, new_annotation_file, model_name="Random Forest"):
        """Evaluate trained model on new data"""
        # Load new data
        new_data = pd.read_csv(new_annotation_file)

        # Engineer features
        # (This should match the feature engineering used in training)
        # Simplified version - you may need to adapt based on available columns

        feature_cols = [
            col
            for col in self.features.columns
            if col != "state" and col in new_data.columns
        ]

        X_new = new_data[feature_cols].fillna(new_data[feature_cols].median())
        y_new = new_data["state"]

        # Get trained model
        model = self.models[model_name]["model"]

        # Predict
        y_pred = model.predict(X_new)

        # Evaluate
        accuracy = accuracy_score(y_new, y_pred)
        print(f"Accuracy on new data: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_new, y_pred))

        return y_pred, accuracy


def main():
    # Example usage
    annotation_files = ["testing/annotations/video-1_annotations_20250922_182616.csv"]

    # Initialize analyzer
    analyzer = AnalysisPipeline(annotation_files)

    # Load data
    data = analyzer.load_and_combine_data()

    # Perform EDA
    summary_stats = analyzer.exploratory_data_analysis()

    # Statistical analysis
    thresholds = analyzer.statistical_threshold_analysis()

    # # Build ML models
    # model_results = analyzer.build_models()

    # # Create rule-based classifier
    # rule_classifier = analyzer.create_rule_based_classifier()

    # # Print best model
    # best_model = max(model_results.items(), key=lambda x: x[1]["test_accuracy"])
    # print(
    #     f"\nBest Model: {best_model[0]} with {best_model[1]['test_accuracy']:.4f} accuracy"
    # )


if __name__ == "__main__":
    main()
