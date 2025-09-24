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

from utilities.general import load_config, load_and_combine_data

warnings.filterwarnings("ignore")


class AnalysisPipeline:
    def __init__(self):
        """
        Initialize analyzer with annotation CSV files

        Args:
            annotation_files: List of paths to annotation CSV files
        """
        self.config = load_config()
        self.data = load_and_combine_data(self.config["annotations"]["output_path"])
        self.features = None
        self.models = {}
        self.thresholds = {}

    def exploratory_data_analysis(self, save_plots=True):
        """Perform comprehensive EDA"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_and_combine_data() first.")

        # Set up the plotting style
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(2, 3, figsize=(18, 15))
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

        # 2. Box plots for distance
        axes[0, 1].set_title("Distance Distribution by State")
        self.data.boxplot(column="mean_distance", by="state", ax=axes[0, 1])
        axes[0, 1].set_xlabel("Rally State")
        axes[0, 1].set_ylabel("Mean Distance")

        # 3. Player positions scatter plot
        axes[0, 2].set_title("Player Positions by State")
        for state in self.data["state"].unique():
            state_data = self.data[self.data["state"] == state]
            valid_data = state_data.dropna(
                subset=["median_player1_x", "median_player1_y"]
            )
            if len(valid_data) > 0:
                axes[0, 2].scatter(
                    valid_data["median_player1_x"],
                    valid_data["median_player1_y"],
                    alpha=0.6,
                    label=f"Player 1 - {state}",
                    s=30,
                )
        axes[0, 2].set_xlabel("Court X Position")
        axes[0, 2].set_ylabel("Court Y Position")
        axes[0, 2].legend()

        # 4. Correlation heatmap
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[numeric_cols].corr()
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", center=0, ax=axes[1, 0], fmt=".2f"
        )
        axes[1, 0].set_title("Feature Correlation Matrix")

        # 5. Summary statistics table
        axes[1, 1].axis("off")
        summary_stats = (
            self.data.groupby("state")[["mean_distance"]]
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

        axes[1, 1].text(
            0.1,
            0.9,
            table_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()
        if save_plots:
            plt.savefig(
                self.config["analysis"]["plots_path"], dpi=300, bbox_inches="tight"
            )
        plt.show()

        return summary_stats

    def statistical_threshold_analysis(self):
        """Analyze statistical thresholds for rule-based classification"""
        print("=== Statistical Threshold Analysis ===")

        results = {}
        for feature in ["mean_distance"]:
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

    def run(self):
        # Perform EDA
        summary_stats = self.exploratory_data_analysis()

        # Statistical analysis
        thresholds = self.statistical_threshold_analysis()


if __name__ == "__main__":
    pipeline = AnalysisPipeline()
    pipeline.run()
