"""
Rally State Segmentation - Testing and Evaluation Module
Provides utilities for evaluating model performance and visualizing results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from modeling.rally_state_segmenter import RallyStateSegmenter
from utilities.general import load_config, load_and_combine_data


class ModelEvaluator:
    """
    Evaluator for rally state segmentation models.
    Provides various metrics and visualization tools.
    """

    def __init__(self, model: RallyStateSegmenter, evaluation_tolerance: int = 5):
        """
        Initialize the evaluator.

        Args:
            model: Trained RallyStateSegmenter model
            evaluation_tolerance: Frame tolerance for lenient evaluation
        """
        self.model = model
        self.evaluation_tolerance = evaluation_tolerance
        self.config = load_config()

    def _create_tolerant_labels(self, labels: np.ndarray, tolerance: int) -> List[set]:
        """
        Create a version of labels where each position can match nearby labels.

        Args:
            labels: Array of state labels
            tolerance: Number of frames tolerance on each side

        Returns:
            List of sets where each set contains valid labels for that position
        """
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
        """
        Calculate accuracy with tolerance for timing errors.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            tolerance: Number of frames tolerance

        Returns:
            Accuracy score with tolerance applied
        """
        tolerant_true = self._create_tolerant_labels(y_true, tolerance)
        correct = sum(1 for i, pred in enumerate(y_pred) if pred in tolerant_true[i])
        return correct / len(y_true)

    def _find_state_transitions(self, states: np.ndarray) -> List[Tuple[int, str, str]]:
        """
        Find all state transitions in a sequence.

        Args:
            states: Array of state labels

        Returns:
            List of tuples (frame_index, from_state, to_state)
        """
        transitions = []
        for i in range(1, len(states)):
            if states[i] != states[i - 1]:
                transitions.append((i, states[i - 1], states[i]))
        return transitions

    def _calculate_transition_accuracy(
        self, y_true: np.ndarray, y_pred: np.ndarray, tolerance: int
    ) -> Dict[str, float]:
        """
        Calculate how well the model detects state transitions.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            tolerance: Frame tolerance for matching transitions

        Returns:
            Dictionary with transition detection metrics
        """
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
        """
        Visualize predictions against ground truth.

        Args:
            df: DataFrame with predictions
            start_frame: Starting frame number
            end_frame: Ending frame number (None for all)
            figsize: Figure size tuple
        """
        if end_frame is None:
            end_frame = len(df)

        df_plot = df.iloc[start_frame:end_frame].copy()

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # Plot 1: Distance with threshold ranges
        ax1 = axes[0]
        ax1.plot(
            df_plot["frame_number"],
            df_plot["mean_distance"],
            label="Raw Distance",
            alpha=0.5,
            linewidth=1,
        )
        if "distance_smoothed" in df_plot.columns:
            ax1.plot(
                df_plot["frame_number"],
                df_plot["distance_smoothed"],
                label="Smoothed Distance",
                linewidth=2,
            )

        # Show range boundaries
        active_min, active_max = tuple(self.model.config["distance_active_range"])
        start_min, start_max = tuple(self.model.config["distance_start_range"])
        end_min, end_max = tuple(self.model.config["distance_end_range"])

        ax1.axhspan(
            active_min, active_max, alpha=0.2, color="green", label="Active Range"
        )
        ax1.axhspan(start_min, start_max, alpha=0.2, color="blue", label="Start Range")
        ax1.axhspan(
            end_min,
            min(end_max, df_plot["mean_distance"].max() * 1.1),
            alpha=0.2,
            color="red",
            label="End Range",
        )

        ax1.set_ylabel("Mean Distance")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Distance Over Time with State Ranges")

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
        """
        Print evaluation metrics in a formatted way.

        Args:
            metrics: Dictionary of metrics from evaluate()
            verbose: Whether to print detailed metrics
        """
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


def main():
    """
    Main execution function for testing and evaluation.
    """
    config = load_config()["annotations"]

    # Load data
    df = load_and_combine_data(config["output_path"])

    print("=" * 60)
    print("RALLY STATE SEGMENTATION - TESTING & EVALUATION")
    print("=" * 60)

    # Data overview
    print("\n1. DATA OVERVIEW")
    print(f"Total frames: {len(df)}")
    print(f"\nState distribution:")
    print(df["state"].value_counts())
    print(f"\nDistance statistics by state:")
    print(df.groupby("state")["mean_distance"].describe())

    model = RallyStateSegmenter()

    # Make predictions
    print("\n3. RUNNING PREDICTIONS...")
    df_pred = model.predict(df, apply_smoothing=True, apply_transitions=True)

    # Evaluate
    print("\n4. EVALUATION RESULTS")
    evaluator = ModelEvaluator(model, evaluation_tolerance=5)
    metrics = evaluator.evaluate(df_pred, use_tolerance=True)
    evaluator.print_metrics(metrics, verbose=True)

    # Visualize
    print("\n5. GENERATING VISUALIZATIONS...")
    evaluator.plot_predictions(df_pred, start_frame=0, end_frame=500)

    return model, df_pred, metrics


if __name__ == "__main__":
    model, df_pred, metrics = main()
