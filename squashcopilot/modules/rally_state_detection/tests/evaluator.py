"""
Rally State Detection Evaluator

Evaluates rally state detection performance on annotated test data.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import glob

from squashcopilot.modules.rally_state_detection import RallyStateDetector
from squashcopilot.common.utils import load_config


class RallyStateEvaluator:
    """Evaluates rally state detection on test videos."""

    def __init__(self, config=None):
        """Initialize evaluator with configuration."""
        if config is None:
            # Load the tests section from the rally_state_detection config
            full_config = load_config(config_name='rally_state_detection')
            config = full_config['tests']
        self.config = config
        self.test_dir = os.path.dirname(os.path.abspath(__file__))

        # Initialize detector
        self.detector = RallyStateDetector()

        # Video properties (set during processing)
        self.video_name = self.config["data"]["video_name"]
        self.data_dir = os.path.join(self.test_dir, self.config["data"]["data_dir"])

    def load_test_data(self) -> pd.DataFrame:
        """
        Load test data from CSV file.

        Returns:
            DataFrame with frame metrics and ground truth states
        """
        print(f"Loading test data for video: {self.video_name}")

        # Load CSV file
        csv_path = glob.glob(os.path.join(self.data_dir, self.video_name, "*.csv"))[0]

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Annotation file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} samples from {csv_path}")

        return df

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Evaluate detector on test data.

        Args:
            df: DataFrame with frame metrics and ground truth states

        Returns:
            Dictionary with evaluation results
        """
        print("Making predictions...")

        # Extract ground truth before prediction
        df_ground_truth = df[["frame_number", "state"]].copy()

        # Remove state column from input to avoid collision during merge
        df_input = df.drop(columns=["state"])

        # Make predictions using the detector
        df_pred = self.detector.process_frames(df_input, aggregated=True)

        # Apply postprocessing
        print("Applying postprocessing...")
        df_pred["postprocessed_state"] = self.detector.postprocess(
            df_pred["predicted_state"]
        )

        # Merge with ground truth
        df_results = pd.merge(df_pred, df_ground_truth, on="frame_number", how="inner")

        # Extract ground truth and predictions
        y_true = df_results["state"]
        y_pred_raw = df_results["predicted_state"]
        y_pred = df_results["postprocessed_state"]

        # Calculate strict frame-level metrics (no tolerance)
        accuracy_raw = accuracy_score(y_true, y_pred_raw)
        accuracy_strict = accuracy_score(y_true, y_pred)

        target_names = ["start", "active", "end"]
        report_strict = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
        )

        cm_strict = confusion_matrix(y_true, y_pred, labels=target_names)

        # Calculate frame-level metrics WITH tolerance
        print("Calculating frame-level metrics with tolerance...")

        y_pred_tolerant = self.apply_tolerance_to_predictions(
            y_true=y_true,
            y_pred=y_pred,
            tolerance_frames=self.config["evaluation"]["tolerance_frames"],
        )

        accuracy_tolerant = accuracy_score(y_true, y_pred_tolerant)
        report_tolerant = classification_report(
            y_true,
            y_pred_tolerant,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        cm_tolerant = confusion_matrix(y_true, y_pred_tolerant, labels=target_names)

        return {
            "accuracy_raw": accuracy_raw,
            "accuracy_strict": accuracy_strict,
            "classification_report_strict": report_strict,
            "confusion_matrix_strict": cm_strict,
            "accuracy_tolerant": accuracy_tolerant,
            "classification_report_tolerant": report_tolerant,
            "confusion_matrix_tolerant": cm_tolerant,
            "y_true": y_true.values,
            "y_pred_raw": y_pred_raw.values,
            "y_pred": y_pred.values,
            "y_pred_tolerant": y_pred_tolerant.values,
            "predictions_df": df_results,
            "tolerance_frames": self.config["evaluation"]["tolerance_frames"],
        }

    def apply_tolerance_to_predictions(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        tolerance_frames: int = 2,
    ) -> pd.Series:
        """
        Apply temporal tolerance to predictions for frame-level evaluation.

        For each frame where prediction doesn't match ground truth, check if
        the ground truth value appears within ±tolerance_frames window. If yes,
        consider it correct by adjusting the prediction.

        Args:
            y_true: Ground truth state sequence
            y_pred: Predicted state sequence
            tolerance_frames: Number of frames to look ahead/behind for tolerance

        Returns:
            Adjusted predictions where mismatches within tolerance are corrected
        """
        y_pred_tolerant = y_pred.copy()

        for i in range(len(y_true)):
            if y_pred.iloc[i] == y_true.iloc[i]:
                # Already correct, no adjustment needed
                continue

            # Check within tolerance window
            window_start = max(0, i - tolerance_frames)
            window_end = min(len(y_true) - 1, i + tolerance_frames)

            # Look for the predicted state in the ground truth window
            predicted_state = y_pred.iloc[i]
            for j in range(window_start, window_end + 1):
                if y_true.iloc[j] == predicted_state:
                    # Found the predicted state within tolerance window
                    # This means the prediction is "close enough", so mark as correct
                    y_pred_tolerant.iloc[i] = y_true.iloc[i]
                    break

        return y_pred_tolerant

    def save_metrics(self, results: dict, output_dir: str):
        """
        Save evaluation metrics to text file.

        Args:
            results: Results dictionary from evaluate()
            output_dir: Output directory path
        """
        output_path = os.path.join(output_dir, "metrics.txt")

        with open(output_path, "w") as f:
            f.write("RALLY STATE DETECTION METRICS\n")
            f.write("=" * 70 + "\n\n")

            # Raw predictions (before postprocessing)
            f.write("RAW PREDICTIONS (Before Postprocessing):\n")
            f.write(f"  Accuracy: {results['accuracy_raw']:.4f}\n\n")

            # Strict metrics (postprocessed, no tolerance)
            f.write("POSTPROCESSED PREDICTIONS (Strict - No Tolerance):\n")
            f.write(f"  Accuracy: {results['accuracy_strict']:.4f}\n\n")

            f.write("  Per-Class Metrics:\n")
            f.write(
                f"  {'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n"
            )
            f.write("  " + "-" * 46 + "\n")
            for class_name in ["start", "active", "end"]:
                if class_name in results["classification_report_strict"]:
                    metrics = results["classification_report_strict"][class_name]
                    f.write(
                        f"  {class_name:<10} {metrics['precision']:.4f}       "
                        f"{metrics['recall']:.4f}       {metrics['f1-score']:.4f}\n"
                    )

            f.write("\n  Confusion Matrix:\n")
            f.write("  (Rows: True, Columns: Predicted)\n")
            cm = results["confusion_matrix_strict"]
            f.write(f"  {'':>10} {'Start':<10} {'Active':<10} {'End':<10}\n")
            for i, state in enumerate(["Start", "Active", "End"]):
                f.write(f"  {state:>10} {cm[i][0]:<10} {cm[i][1]:<10} {cm[i][2]:<10}\n")

            # Tolerant metrics
            f.write(f"\n\nWITH TOLERANCE (±{results['tolerance_frames']} frames):\n")
            f.write(f"  Accuracy: {results['accuracy_tolerant']:.4f}\n\n")

            f.write("  Per-Class Metrics:\n")
            f.write(
                f"  {'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n"
            )
            f.write("  " + "-" * 46 + "\n")
            for class_name in ["start", "active", "end"]:
                if class_name in results["classification_report_tolerant"]:
                    metrics = results["classification_report_tolerant"][class_name]
                    f.write(
                        f"  {class_name:<10} {metrics['precision']:.4f}       "
                        f"{metrics['recall']:.4f}       {metrics['f1-score']:.4f}\n"
                    )

            f.write("\n  Confusion Matrix:\n")
            f.write("  (Rows: True, Columns: Predicted)\n")
            cm = results["confusion_matrix_tolerant"]
            f.write(f"  {'':>10} {'Start':<10} {'Active':<10} {'End':<10}\n")
            for i, state in enumerate(["Start", "Active", "End"]):
                f.write(f"  {state:>10} {cm[i][0]:<10} {cm[i][1]:<10} {cm[i][2]:<10}\n")

    def save_plot(self, results: dict, output_dir: str):
        """
        Save prediction visualization plot.

        Args:
            results: Results dictionary from evaluate()
            output_dir: Output directory path
        """
        output_path = os.path.join(output_dir, "predictions_plot.png")
        dpi = self.config["output"]["plot_dpi"]

        df = results["predictions_df"]

        # Create state mapping for plotting
        state_map = {"start": 0, "active": 1, "end": 2}

        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

        # Plot 1: Ground truth states
        y_true_numeric = [state_map[state] for state in df["state"]]
        axes[0].fill_between(
            df["frame_number"],
            0,
            y_true_numeric,
            alpha=0.7,
            step="post",
            label="Ground Truth",
            color="blue",
        )
        axes[0].set_ylabel("State", fontsize=12)
        axes[0].set_yticks([0, 1, 2])
        axes[0].set_yticklabels(["Start", "Active", "End"])
        axes[0].set_title("Ground Truth States", fontsize=14, fontweight="bold")
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Raw Predicted states (before postprocessing)
        y_pred_raw_numeric = [state_map[state] for state in df["predicted_state"]]
        axes[1].fill_between(
            df["frame_number"],
            0,
            y_pred_raw_numeric,
            alpha=0.7,
            step="post",
            color="orange",
            label="Raw Predictions",
        )
        axes[1].set_ylabel("State", fontsize=12)
        axes[1].set_yticks([0, 1, 2])
        axes[1].set_yticklabels(["Start", "Active", "End"])
        axes[1].set_title(
            f"Raw Predicted States (Accuracy: {results['accuracy_raw']:.2%})",
            fontsize=14,
            fontweight="bold",
        )
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Postprocessed Predicted states
        y_pred_numeric = [state_map[state] for state in df["postprocessed_state"]]
        axes[2].fill_between(
            df["frame_number"],
            0,
            y_pred_numeric,
            alpha=0.7,
            step="post",
            color="green",
            label="Postprocessed",
        )
        axes[2].set_ylabel("State", fontsize=12)
        axes[2].set_yticks([0, 1, 2])
        axes[2].set_yticklabels(["Start", "Active", "End"])
        axes[2].set_xlabel("Frame Number", fontsize=12)
        axes[2].set_title(
            f"Postprocessed Predicted States (Accuracy: {results['accuracy_strict']:.2%})",
            fontsize=14,
            fontweight="bold",
        )
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def print_results(self, results: dict):
        """Print evaluation results."""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)

        # Raw predictions
        print("\nRAW PREDICTIONS (Before Postprocessing):")
        print(f"  Accuracy: {results['accuracy_raw']:.4f}")

        # Frame-level metrics (STRICT)
        print("\nPOSTPROCESSED PREDICTIONS (Strict - No Tolerance):")
        print(f"  Accuracy: {results['accuracy_strict']:.4f}")

        print("\n  Per-Class Metrics:")
        print(f"  {'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("  " + "-" * 46)

        for class_name in ["start", "active", "end"]:
            if class_name in results["classification_report_strict"]:
                metrics = results["classification_report_strict"][class_name]
                print(
                    f"  {class_name:<10} {metrics['precision']:.4f}       "
                    f"{metrics['recall']:.4f}       {metrics['f1-score']:.4f}"
                )

        print("\n  Confusion Matrix:")
        print("  (Rows: True, Columns: Predicted)")
        cm = results["confusion_matrix_strict"]
        print(f"  {'':>10} {'Start':<10} {'Active':<10} {'End':<10}")
        for i, state in enumerate(["Start", "Active", "End"]):
            print(f"  {state:>10} {cm[i][0]:<10} {cm[i][1]:<10} {cm[i][2]:<10}")

        # Frame-level metrics (WITH TOLERANCE)
        print(f"\n\nWITH TOLERANCE (±{results['tolerance_frames']} frames):")
        print(f"  Accuracy: {results['accuracy_tolerant']:.4f}")

        print("\n  Per-Class Metrics:")
        print(f"  {'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("  " + "-" * 46)

        for class_name in ["start", "active", "end"]:
            if class_name in results["classification_report_tolerant"]:
                metrics = results["classification_report_tolerant"][class_name]
                print(
                    f"  {class_name:<10} {metrics['precision']:.4f}       "
                    f"{metrics['recall']:.4f}       {metrics['f1-score']:.4f}"
                )

        print("\n  Confusion Matrix:")
        print("  (Rows: True, Columns: Predicted)")
        cm = results["confusion_matrix_tolerant"]
        print(f"  {'':>10} {'Start':<10} {'Active':<10} {'End':<10}")
        for i, state in enumerate(["Start", "Active", "End"]):
            print(f"  {state:>10} {cm[i][0]:<10} {cm[i][1]:<10} {cm[i][2]:<10}")

    def save_results(self, results: dict, output_dir: str):
        """
        Save all results.

        Args:
            results: Results dictionary from evaluate()
            output_dir: Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving results to: {output_dir}")

        # Save metrics
        self.save_metrics(results, output_dir)
        print("  ✓ Metrics saved")

        # Save plot
        self.save_plot(results, output_dir)
        print("  ✓ Plot saved")

    def run_evaluation(self) -> dict:
        """
        Run complete evaluation pipeline.

        Returns:
            Dictionary with evaluation results
        """
        print("=" * 70)
        print("RALLY STATE DETECTION EVALUATION")
        print("=" * 70)

        # Load test data
        df = self.load_test_data()

        # Evaluate
        results = self.evaluate(df)

        # Print results
        self.print_results(results)

        # Save results
        output_dir = os.path.join(
            self.test_dir,
            self.config["output"]["output_dir"],
            self.video_name,
        )
        self.save_results(results, output_dir)

        print("\n" + "=" * 70)
        print(f"✓ Evaluation complete!")
        print(f"✓ Results saved to: {output_dir}")
        print("=" * 70)

        return results


def main():
    """Main entry point."""
    evaluator = RallyStateEvaluator()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
