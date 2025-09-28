"""
Evaluator - Evaluate trained models on test data
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import glob

from modeling.predictor import StatePredictor
from config import CONFIG


class ModelEvaluator:
    """Evaluates trained models on test data."""

    def __init__(self):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model
        """
        self.video_filter = CONFIG["evaluator"]["video_filter"]
        self.predictor = StatePredictor()

    def load_test_data(self) -> pd.DataFrame:
        """
        Load test data from CSV files.

        Args:
            video_filter: If provided, only load this specific video

        Returns:
            DataFrame with base metrics and ground truth states
        """
        print("Loading test data...")

        # Find CSV files
        csv_files = glob.glob(os.path.join(CONFIG["annotations"]["data_path"], "*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {CONFIG["annotations"]['data_path']}"
            )

        # Load data
        dataframes = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)

            # Add video name if not present
            if "video_name" not in df.columns:
                video_name = os.path.splitext(os.path.basename(csv_file))[0]
                df["video_name"] = video_name

            # Filter to specific video if requested
            if (
                self.video_filter is None
                or df["video_name"].iloc[0] == self.video_filter
            ):
                dataframes.append(df)
                print(f"  Loaded {len(df)} samples from {csv_file}")

        if not dataframes:
            raise ValueError(f"No data found for video filter: {self.video_filter}")

        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Test dataset: {len(combined_df)} samples")

        return combined_df

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Evaluate model on test data.

        Args:
            df: DataFrame with base metrics and ground truth states

        Returns:
            Dictionary with evaluation metrics
        """
        print("Making predictions...")

        # Make predictions using the predictor
        df_pred = self.predictor.predict_batch(df)

        # Extract ground truth and predictions
        y_true = df_pred["state"].values
        y_pred = df_pred["predicted_state"].values

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Classification report
        target_names = ["start", "active", "end"]
        report = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=target_names)

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm,
            "y_true": y_true,
            "y_pred": y_pred,
            "predictions_df": df_pred,
        }

    def print_results(self, results: dict):
        """Print evaluation results."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        print(f"\nOverall Accuracy: {results['accuracy']:.4f}")

        print("\nPer-Class Metrics:")
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 46)

        for class_name in ["start", "active", "end"]:
            if class_name in results["classification_report"]:
                metrics = results["classification_report"][class_name]
                print(
                    f"{class_name:<10} {metrics['precision']:.3f}        "
                    f"{metrics['recall']:.3f}        {metrics['f1-score']:.3f}"
                )

        print("\nConfusion Matrix:")
        print("(Rows: True, Columns: Predicted)")
        cm = results["confusion_matrix"]
        print(f"{'':>10} {'Start':<10} {'Active':<10} {'End':<10}")
        for i, state in enumerate(["Start", "Active", "End"]):
            print(f"{state:>10} {cm[i][0]:<10} {cm[i][1]:<10} {cm[i][2]:<10}")

    def plot_predictions(self, results: dict, start_idx: int = 0, end_idx: int = None):
        """
        Plot predictions vs ground truth and save to config path.

        Args:
            results: Results from evaluate()
            start_idx: Start index for plotting
            end_idx: End index for plotting (None for all data)
        """
        df = results["predictions_df"]

        if end_idx is None:
            end_idx = len(df)

        df_plot = df.iloc[start_idx:end_idx]

        # Create state mapping for plotting
        state_map = {"start": 0, "active": 1, "end": 2}

        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

        # Plot 1: Distance over time
        axes[0].plot(
            df_plot.index, df_plot["mean_distance"], "b-", alpha=0.7, linewidth=2
        )
        axes[0].set_ylabel("Mean Distance")
        axes[0].set_title("Distance Over Time")
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Ground truth states
        y_true_numeric = [state_map[state] for state in df_plot["state"]]
        axes[1].fill_between(
            df_plot.index,
            0,
            y_true_numeric,
            alpha=0.7,
            step="post",
            label="Ground Truth",
        )
        axes[1].set_ylabel("State")
        axes[1].set_yticks([0, 1, 2])
        axes[1].set_yticklabels(["Start", "Active", "End"])
        axes[1].set_title("Ground Truth States")
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Predicted states
        y_pred_numeric = [state_map[state] for state in df_plot["predicted_state"]]
        axes[2].fill_between(
            df_plot.index,
            0,
            y_pred_numeric,
            alpha=0.7,
            step="post",
            color="green",
            label="Predicted",
        )
        axes[2].set_ylabel("State")
        axes[2].set_yticks([0, 1, 2])
        axes[2].set_yticklabels(["Start", "Active", "End"])
        axes[2].set_xlabel("Sample Index")
        axes[2].set_title("Predicted States")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot to config path
        plot_path = CONFIG["evaluator"]["plot_output_path"]

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

        # Save the plot
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {plot_path}")

        # Optionally still show the plot (comment out if you don't want to display)
        plt.show()

        # Close the figure to free memory
        plt.close(fig)

        return fig

    def run_evaluation(self) -> dict:
        """
        Run complete evaluation pipeline.

        Args:
            video_filter: Optional video name to filter evaluation to

        Returns:
            Dictionary with evaluation results
        """
        print("=" * 60)
        print("MODEL EVALUATION PIPELINE")
        print("=" * 60)

        # Load test data
        df = self.load_test_data()

        # Evaluate
        results = self.evaluate(df)

        # Print results
        self.print_results(results)

        # Plot results
        print("\nGenerating visualization...")
        self.plot_predictions(results, end_idx=min(1000, len(df)))

        return results


if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    results = evaluator.run_evaluation()
