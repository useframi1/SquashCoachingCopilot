"""
Clean Model Evaluator Pipeline
Model-agnostic evaluation using the unified prediction interface.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from modeling.unified_predictor import UnifiedPredictor
from utilities.general import load_config, load_and_combine_data


class ModelEvaluator:
    """
    Clean, model-agnostic evaluator for rally state prediction.
    Uses unified predictor interface for consistency.
    """

    def __init__(self, evaluation_tolerance: int = 5):
        """
        Initialize the evaluator.

        Args:
            evaluation_tolerance: Frame tolerance for lenient evaluation
        """
        self.evaluation_tolerance = evaluation_tolerance
        self.config = load_config()

        # Initialize unified predictor
        self.predictor = UnifiedPredictor()
        print(f"Initialized evaluator with: {self.predictor.get_model_info()}")

    def _create_tolerant_labels(self, labels: np.ndarray, tolerance: int) -> List[set]:
        """Create tolerant labels for evaluation with timing tolerance."""
        n = len(labels)
        tolerant_labels = []

        for i in range(n):
            start_idx = max(0, i - tolerance)
            end_idx = min(n, i + tolerance + 1)
            window_labels = set(labels[start_idx:end_idx])
            tolerant_labels.append(window_labels)

        return tolerant_labels

    def _calculate_tolerant_accuracy(
        self, y_true: np.ndarray, y_pred: np.ndarray, tolerance: int
    ) -> float:
        """Calculate accuracy with tolerance for timing errors."""
        tolerant_true = self._create_tolerant_labels(y_true, tolerance)
        correct = sum(1 for i, pred in enumerate(y_pred) if pred in tolerant_true[i])
        return correct / len(y_true)

    def _find_state_transitions(self, states: np.ndarray) -> List[Tuple[int, str, str]]:
        """Find all state transitions in a sequence."""
        transitions = []
        for i in range(1, len(states)):
            if states[i] != states[i - 1]:
                transitions.append((i, states[i - 1], states[i]))
        return transitions

    def _calculate_transition_accuracy(
        self, y_true: np.ndarray, y_pred: np.ndarray, tolerance: int
    ) -> Dict[str, float]:
        """Calculate transition detection metrics."""
        true_transitions = self._find_state_transitions(y_true)
        pred_transitions = self._find_state_transitions(y_pred)

        if len(true_transitions) == 0:
            return {
                "transition_precision": 1.0,
                "transition_recall": 1.0,
                "avg_transition_error": 0.0,
                "num_true_transitions": 0,
                "num_pred_transitions": len(pred_transitions),
                "num_matched_transitions": 0,
            }

        matched_transitions = 0
        transition_errors = []

        for true_idx, true_from, true_to in true_transitions:
            matches = [
                (pred_idx, abs(pred_idx - true_idx))
                for pred_idx, pred_from, pred_to in pred_transitions
                if pred_from == true_from
                and pred_to == true_to
                and abs(pred_idx - true_idx) <= tolerance
            ]

            if matches:
                matched_transitions += 1
                _, error = min(matches, key=lambda x: x[1])
                transition_errors.append(error)

        recall = matched_transitions / len(true_transitions) if true_transitions else 0
        precision = (
            matched_transitions / len(pred_transitions) if pred_transitions else 0
        )
        avg_error = np.mean(transition_errors) if transition_errors else 0

        return {
            "transition_precision": precision,
            "transition_recall": recall,
            "avg_transition_error": avg_error,
            "num_true_transitions": len(true_transitions),
            "num_pred_transitions": len(pred_transitions),
            "num_matched_transitions": matched_transitions,
        }

    def make_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on DataFrame using unified predictor.

        Args:
            df: DataFrame with base metrics (from MetricsAggregator)

        Returns:
            DataFrame with predictions added
        """
        df_pred = df.copy()

        # Reset predictor state for new evaluation
        self.predictor.reset_state()

        # Use unified predictor for consistent results
        predictions = self.predictor.predict(df_pred)
        df_pred["predicted_state"] = predictions

        return df_pred

    def evaluate(
        self, df: pd.DataFrame, use_tolerance: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model performance against ground truth.

        Args:
            df: DataFrame with 'state' and 'predicted_state' columns
            use_tolerance: Whether to apply frame tolerance in evaluation

        Returns:
            Dictionary of performance metrics
        """
        y_true = df["state"].values
        y_pred = df["predicted_state"].values

        # Standard metrics (strict, frame-by-frame)
        strict_accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=["start", "active", "end"], zero_division=0
        )

        cm = confusion_matrix(y_true, y_pred, labels=["start", "active", "end"])

        metrics = {
            "strict_accuracy": strict_accuracy,
            "precision_start": precision[0],
            "precision_active": precision[1],
            "precision_end": precision[2],
            "recall_start": recall[0],
            "recall_active": recall[1],
            "recall_end": recall[2],
            "f1_start": f1[0],
            "f1_active": f1[1],
            "f1_end": f1[2],
            "confusion_matrix": cm,
        }

        # Add tolerant metrics if requested
        if use_tolerance:
            tolerant_accuracy = self._calculate_tolerant_accuracy(
                y_true, y_pred, self.evaluation_tolerance
            )
            metrics["tolerant_accuracy"] = tolerant_accuracy

            transition_metrics = self._calculate_transition_accuracy(
                y_true, y_pred, self.evaluation_tolerance
            )
            metrics.update(transition_metrics)

        return metrics

    def plot_predictions(
        self,
        df: pd.DataFrame,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        figsize: Tuple[int, int] = (15, 8),
    ):
        """Visualize predictions against ground truth."""
        if end_frame is None:
            end_frame = len(df)

        df_plot = df.iloc[start_frame:end_frame].copy()

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # Plot 1: Distance over time
        ax1 = axes[0]
        ax1.plot(
            df_plot["frame_number"],
            df_plot["mean_distance"],
            label="Mean Distance",
            alpha=0.7,
            linewidth=2,
        )

        ax1.set_ylabel("Mean Distance")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        model_info = self.predictor.get_model_info()
        ax1.set_title(
            f"Distance Over Time ({model_info['model_type'].replace('_', ' ').title()})"
        )

        # Plot 2: Ground truth states
        ax2 = axes[1]
        state_map = {"start": 0, "active": 1, "end": 2}
        df_plot["state_numeric"] = df_plot["state"].map(state_map)

        ax2.fill_between(
            df_plot["frame_number"],
            0,
            df_plot["state_numeric"],
            alpha=0.5,
            label="Ground Truth",
            step="post",
        )
        ax2.set_ylabel("State")
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(["Start", "Active", "End"])
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Ground Truth States")

        # Plot 3: Predicted states
        ax3 = axes[2]
        df_plot["predicted_numeric"] = df_plot["predicted_state"].map(state_map)

        ax3.fill_between(
            df_plot["frame_number"],
            0,
            df_plot["predicted_numeric"],
            alpha=0.5,
            color="green",
            label="Predicted",
            step="post",
        )
        ax3.set_ylabel("State")
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(["Start", "Active", "End"])
        ax3.set_xlabel("Frame Number")
        ax3.legend(loc="upper right")
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Predicted States")

        plt.tight_layout()
        plt.show()

        return fig

    def print_metrics(self, metrics: Dict[str, float], verbose: bool = True):
        """Print evaluation metrics in a formatted way."""
        model_info = self.predictor.get_model_info()
        print(
            f"\n=== EVALUATION RESULTS ({model_info['model_type'].replace('_', ' ').title()}) ==="
        )

        print(f"\nStrict Accuracy (frame-by-frame): {metrics['strict_accuracy']:.3f}")

        if "tolerant_accuracy" in metrics:
            print(
                f"Tolerant Accuracy (±{self.evaluation_tolerance} frames): {metrics['tolerant_accuracy']:.3f}"
            )
            improvement = (
                metrics["tolerant_accuracy"] - metrics["strict_accuracy"]
            ) * 100
            print(f"  → Improvement: +{improvement:.1f}%")

        if verbose:
            print("\nPer-State Metrics (Strict):")
            print(f"{'State':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
            print("-" * 46)
            for state in ["start", "active", "end"]:
                print(
                    f"{state:<10} {metrics[f'precision_{state}']:.3f}        "
                    f"{metrics[f'recall_{state}']:.3f}        "
                    f"{metrics[f'f1_{state}']:.3f}"
                )

        if "transition_precision" in metrics:
            print("\nTransition Detection Metrics:")
            print(
                f"  Precision: {metrics['transition_precision']:.3f} "
                f"({metrics['num_matched_transitions']}/{metrics['num_pred_transitions']} predicted transitions correct)"
            )
            print(
                f"  Recall: {metrics['transition_recall']:.3f} "
                f"({metrics['num_matched_transitions']}/{metrics['num_true_transitions']} true transitions detected)"
            )
            print(f"  Avg Error: {metrics['avg_transition_error']:.1f} frames")

        if verbose:
            print("\nConfusion Matrix:")
            print("(Rows: True, Columns: Predicted)")
            print(f"{'':>10} {'Start':<10} {'Active':<10} {'End':<10}")
            cm = metrics["confusion_matrix"]
            for i, state in enumerate(["Start", "Active", "End"]):
                print(f"{state:>10} {cm[i][0]:<10} {cm[i][1]:<10} {cm[i][2]:<10}")

    def run(self, video_filter: Optional[str] = None):
        """
        Run complete model evaluation pipeline.

        Args:
            video_filter: Optional video name to filter (e.g., "video-2")

        Returns:
            Tuple of (df_pred, metrics)
        """
        # Load data
        df = load_and_combine_data(self.config["annotations"]["output_path"])

        if video_filter:
            df = df[df["video_name"] == video_filter]
            print(f"Filtered to video: {video_filter}")

        print("=" * 60)
        print("RALLY STATE SEGMENTATION - MODEL EVALUATION")
        print("=" * 60)

        # Data overview
        print("\n1. DATA OVERVIEW")
        print(f"Total frames: {len(df)}")
        print(f"\nState distribution:")
        print(df["state"].value_counts())
        print(f"\nVideos in dataset: {df['video_name'].nunique()}")
        print(f"Video names: {sorted(df['video_name'].unique())}")

        # Make predictions
        print("\n2. RUNNING PREDICTIONS...")
        df_pred = self.make_predictions(df)

        # Save predictions
        output_file = f"testing/predictions_{self.predictor.model_type}.csv"
        df_pred.to_csv(output_file, index=False)
        print(f"Predictions saved to: {output_file}")

        # Evaluate
        print("\n3. EVALUATION RESULTS")
        metrics = self.evaluate(df_pred, use_tolerance=True)
        self.print_metrics(metrics, verbose=True)

        # Visualize
        print("\n4. GENERATING VISUALIZATIONS...")
        self.plot_predictions(df_pred, start_frame=0, end_frame=min(500, len(df_pred)))

        return df_pred, metrics


if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator(evaluation_tolerance=5)
    df_pred, metrics = evaluator.run(video_filter="video-2")
