"""
Rally State Detection Model Training

This module provides functionality to train an LSTM model for rally state detection
using annotated video data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from squashcopilot.common.utils import load_config
from squashcopilot.modules.rally_state_detection.models.lstm_model import (
    RallyStateLSTM,
)


class RallyStateDataset(Dataset):
    """PyTorch Dataset for rally state detection sequences."""

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Initialize the dataset.

        Args:
            sequences: Array of shape (num_sequences, sequence_length, num_features)
            labels: Array of shape (num_sequences, sequence_length)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels).unsqueeze(-1)  # Add dimension for BCE

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class RallyStateTrainer:
    """Trainer class for LSTM-based rally state detection."""

    def __init__(self, config_name: str = "rally_state_detection"):
        """
        Initialize the trainer.

        Args:
            config_name: Name of the configuration file (without .yaml extension)
        """
        self.config = load_config(config_name=config_name)
        self.training_config = self.config["training"]

        # Get project root
        project_root = Path(__file__).parent.parent.parent.parent
        self.data_dir = project_root / self.training_config["data_dir"]
        self.model_save_path = project_root / self.training_config["model_save_path"]
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        # self.annotations_dir = project_root / self.training_config.get(
        #     "annotations_dir", "squashcopilot/annotation/annotations"
        # )

        # Extract configuration
        self.features = self.training_config["features"]
        self.label_column = self.training_config["label_column"]
        self.label_mapping = self.training_config["labels"]
        self.sequence_length = self.training_config["sequence_length"]
        self.train_test_split = self.training_config["train_test_split"]
        self.validation_split = self.training_config["validation_split"]
        self.batch_size = self.training_config["batch_size"]
        self.epochs = self.training_config["epochs"]
        self.learning_rate = self.training_config["learning_rate"]

        # Performance configuration
        self.num_workers = self.training_config.get("num_workers", 0)

        # Early stopping configuration
        early_stopping_config = self.training_config.get("early_stopping", {})
        self.early_stopping_enabled = early_stopping_config.get("enabled", False)
        self.early_stopping_patience = early_stopping_config.get("patience", 10)
        self.early_stopping_min_delta = early_stopping_config.get("min_delta", 0.0001)

        # Model configuration
        model_config = self.training_config["model"]
        self.hidden_size = model_config["hidden_size"]
        self.num_layers = model_config["num_layers"]
        self.dropout = model_config["dropout"]
        self.bidirectional = model_config["bidirectional"]

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configure multi-threading for CPU operations
        # PyTorch will use multiple CPU cores for matrix operations
        num_threads = self.training_config.get("num_threads", None)
        if num_threads is not None:
            torch.set_num_threads(num_threads)
            print(f"Using device: {self.device} with {num_threads} threads")
        else:
            # Use PyTorch's default (usually all available cores)
            print(f"Using device: {self.device} with default threading")
            print(f"Available CPU cores: {torch.get_num_threads()}")

        # Initialize model, criterion, optimizer
        self.model = None
        self.criterion = nn.BCELoss()
        self.optimizer = None

        # Data containers
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def _load_video_data(self, video_path: Path) -> pd.DataFrame:
        """
        Load data from a single video CSV file by joining test data with annotations.

        Args:
            video_path: Path to the video directory in tests/data

        Returns:
            DataFrame containing the merged video data with required features
        """
        # Load test data CSV
        test_csv = video_path / f"{video_path.name}_annotations.csv"
        if not test_csv.exists():
            raise FileNotFoundError(f"Test CSV file not found: {test_csv}")

        # # Load annotation CSV
        # annotation_csv = (
        #     self.annotations_dir
        #     / video_path.name
        #     / f"{video_path.name}_annotations.csv"
        # )
        # if not annotation_csv.exists():
        #     raise FileNotFoundError(f"Annotation CSV file not found: {annotation_csv}")

        # Read both CSVs
        df = pd.read_csv(test_csv)
        # annotation_df = pd.read_csv(annotation_csv)

        # # Inner join on frame column
        # df = pd.merge(
        #     test_df,
        #     annotation_df,
        #     on="frame",
        #     how="inner",
        # )

        # Select only required features and label column
        # Prefer columns from test data if they exist in both
        # required_columns = ["frame"] + self.features + [self.label_column]

        # Keep only required columns
        # df = df[required_columns]

        # Verify all required features exist
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features after merge: {missing_features}")

        if self.label_column not in df.columns:
            raise ValueError(
                f"Label column '{self.label_column}' not found after merge"
            )

        return df

    def _create_sequences(
        self, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create overlapping sequences from video data.

        Args:
            features: Feature array of shape (num_frames, num_features)
            labels: Label array of shape (num_frames,)

        Returns:
            Tuple of (sequences, sequence_labels)
            - sequences: shape (num_sequences, sequence_length, num_features)
            - sequence_labels: shape (num_sequences, sequence_length)
        """
        num_frames = len(features)
        sequences = []
        sequence_labels = []

        # Create sliding window sequences
        for i in range(num_frames - self.sequence_length + 1):
            seq = features[i : i + self.sequence_length]
            seq_labels = labels[i : i + self.sequence_length]
            sequences.append(seq)
            sequence_labels.append(seq_labels)

        return np.array(sequences), np.array(sequence_labels)

    def load_data(self) -> None:
        """
        Load and prepare all video data for training.

        Creates train, validation, and test datasets with proper video-level splits.
        """
        print("=" * 60)
        print("Loading Data")
        print("=" * 60)

        # Get all video directories
        video_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        print(f"Found {len(video_dirs)} videos")

        # Split videos into train and test sets
        num_train = int(len(video_dirs) * self.train_test_split)
        train_video_dirs = video_dirs[:num_train]
        test_video_dirs = video_dirs[num_train:]

        print(f"Train videos: {len(train_video_dirs)}")
        print(f"Test videos: {len(test_video_dirs)}")

        # Load and process training videos
        train_sequences = []
        train_labels = []

        for video_dir in train_video_dirs:
            print(f"Loading training video: {video_dir.name}")
            df = self._load_video_data(video_dir)

            # Extract features
            features = df[self.features].values

            # Map labels to binary (start -> 1, end -> 0)
            labels = df[self.label_column].map(self.label_mapping).values

            # Create sequences
            seqs, seq_labels = self._create_sequences(features, labels)
            train_sequences.append(seqs)
            train_labels.append(seq_labels)

            print(f"  - Created {len(seqs)} sequences")

        # Concatenate all training sequences
        train_sequences = np.concatenate(train_sequences, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        print(f"\nTotal training sequences: {len(train_sequences)}")

        # Create validation split if specified
        if self.validation_split > 0:
            num_val = int(len(train_sequences) * self.validation_split)
            val_sequences = train_sequences[:num_val]
            val_labels = train_labels[:num_val]
            train_sequences = train_sequences[num_val:]
            train_labels = train_labels[num_val:]

            print(f"Validation sequences: {len(val_sequences)}")
            print(f"Training sequences (after split): {len(train_sequences)}")

            # Create validation dataset and loader
            val_dataset = RallyStateDataset(val_sequences, val_labels)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True if self.device.type == "cuda" else False,
            )
        else:
            self.val_loader = None
            print("No validation split")

        # Create training dataset and loader
        train_dataset = RallyStateDataset(train_sequences, train_labels)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # Load and process test videos
        test_sequences = []
        test_labels = []

        for video_dir in test_video_dirs:
            print(f"Loading test video: {video_dir.name}")
            df = self._load_video_data(video_dir)

            # Extract features
            features = df[self.features].values

            # Map labels to binary
            labels = df[self.label_column].map(self.label_mapping).values

            # Create sequences
            seqs, seq_labels = self._create_sequences(features, labels)
            test_sequences.append(seqs)
            test_labels.append(seq_labels)

            print(f"  - Created {len(seqs)} sequences")

        # Concatenate all test sequences
        test_sequences = np.concatenate(test_sequences, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        print(f"\nTotal test sequences: {len(test_sequences)}")

        # Create test dataset and loader
        test_dataset = RallyStateDataset(test_sequences, test_labels)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        print("=" * 60)
        print(f"Feature dimensions: {len(self.features)}")
        print(f"Sequence length: {self.sequence_length}")
        print("=" * 60)

    def _initialize_model(self) -> None:
        """Initialize the LSTM model."""
        input_size = len(self.features)
        self.model = RallyStateLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Print model info
        model_info = self.model.get_model_info()
        print("\nModel Architecture:")
        print(f"  Input size: {model_info['input_size']}")
        print(f"  Hidden size: {model_info['hidden_size']}")
        print(f"  Num layers: {model_info['num_layers']}")
        print(f"  Bidirectional: {model_info['bidirectional']}")
        print(f"  Total parameters: {model_info['total_parameters']:,}")
        print(f"  Trainable parameters: {model_info['trainable_parameters']:,}")

    def _compute_metrics(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            predictions: Binary predictions
            labels: Ground truth labels

        Returns:
            Dictionary of metrics
        """
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def _train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []

        # Create progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs} [Train]",
            leave=False,
        )

        for sequences, labels in pbar:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Collect predictions for metrics
            predictions = (outputs > 0.5).float()
            all_predictions.append(predictions.cpu().numpy().flatten())
            all_labels.append(labels.cpu().numpy().flatten())

            # Update progress bar with current loss
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Compute metrics
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        metrics = self._compute_metrics(all_predictions, all_labels)

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, metrics

    def _validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.

        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        if self.val_loader is None:
            return 0.0, {}

        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        # Create progress bar
        pbar = tqdm(
            self.val_loader,
            desc="Validation",
            leave=False,
        )

        with torch.no_grad():
            for sequences, labels in pbar:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                predictions = (outputs > 0.5).float()
                all_predictions.append(predictions.cpu().numpy().flatten())
                all_labels.append(labels.cpu().numpy().flatten())

                # Update progress bar with current loss
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Compute metrics
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        metrics = self._compute_metrics(all_predictions, all_labels)

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, metrics

    def train(self) -> None:
        """Train the model with optional early stopping."""
        print("\n" + "=" * 60)
        print("Training Model")
        print("=" * 60)

        if self.early_stopping_enabled:
            print(
                f"Early stopping enabled: patience={self.early_stopping_patience}, "
                f"min_delta={self.early_stopping_min_delta}"
            )

        # Initialize model
        self._initialize_model()

        best_val_f1 = 0.0
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")
            print("-" * 60)

            # Train
            train_loss, train_metrics = self._train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")
            print(
                f"Train - Acc: {train_metrics['accuracy']:.4f}, "
                f"Prec: {train_metrics['precision']:.4f}, "
                f"Rec: {train_metrics['recall']:.4f}, "
                f"F1: {train_metrics['f1']:.4f}"
            )

            # Validate
            if self.val_loader is not None:
                val_loss, val_metrics = self._validate()
                print(f"Val Loss: {val_loss:.4f}")
                print(
                    f"Val - Acc: {val_metrics['accuracy']:.4f}, "
                    f"Prec: {val_metrics['precision']:.4f}, "
                    f"Rec: {val_metrics['recall']:.4f}, "
                    f"F1: {val_metrics['f1']:.4f}"
                )

                # Check for improvement
                improvement = val_metrics["f1"] - best_val_f1

                # Save best model based on validation F1
                if improvement > self.early_stopping_min_delta:
                    best_val_f1 = val_metrics["f1"]
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    self._save_model("best")
                    print(f"  --> New best model! (F1: {best_val_f1:.4f})")
                else:
                    epochs_without_improvement += 1
                    if self.early_stopping_enabled:
                        print(
                            f"  --> No improvement for {epochs_without_improvement} epoch(s)"
                        )

                # Early stopping check
                if (
                    self.early_stopping_enabled
                    and epochs_without_improvement >= self.early_stopping_patience
                ):
                    print(f"\n{'=' * 60}")
                    print(f"Early stopping triggered after {epoch} epochs")
                    print(
                        f"No improvement for {self.early_stopping_patience} consecutive epochs"
                    )
                    print(f"Best F1: {best_val_f1:.4f} at epoch {best_epoch}")
                    print(f"{'=' * 60}")
                    break

            else:
                # No validation, save based on training F1
                improvement = train_metrics["f1"] - best_val_f1

                if improvement > self.early_stopping_min_delta:
                    best_val_f1 = train_metrics["f1"]
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    self._save_model("best")
                    print(f"  --> New best model! (F1: {best_val_f1:.4f})")
                else:
                    epochs_without_improvement += 1
                    if self.early_stopping_enabled:
                        print(
                            f"  --> No improvement for {epochs_without_improvement} epoch(s)"
                        )

                # Early stopping check (for training F1 when no validation)
                if (
                    self.early_stopping_enabled
                    and epochs_without_improvement >= self.early_stopping_patience
                ):
                    print(f"\n{'=' * 60}")
                    print(f"Early stopping triggered after {epoch} epochs")
                    print(
                        f"No improvement for {self.early_stopping_patience} consecutive epochs"
                    )
                    print(f"Best F1: {best_val_f1:.4f} at epoch {best_epoch}")
                    print(f"{'=' * 60}")
                    break

        print("\n" + "=" * 60)
        print(f"Training completed!")
        print(f"Best model from epoch {best_epoch} with F1: {best_val_f1:.4f}")
        print("=" * 60)

        # Save final model
        self._save_model("final")

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on test set.

        Returns:
            Dictionary of test metrics
        """
        print("\n" + "=" * 60)
        print("Evaluating on Test Set")
        print("=" * 60)

        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in self.test_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(sequences)
                predictions = (outputs > 0.5).float()

                all_predictions.append(predictions.cpu().numpy().flatten())
                all_labels.append(labels.cpu().numpy().flatten())

        # Compute metrics
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        metrics = self._compute_metrics(all_predictions, all_labels)

        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test Precision: {metrics['precision']:.4f}")
        print(f"Test Recall: {metrics['recall']:.4f}")
        print(f"Test F1-Score: {metrics['f1']:.4f}")
        print("=" * 60)

        return metrics

    def _save_model(self, suffix: str = "final") -> None:
        """
        Save the model checkpoint.

        Args:
            suffix: Suffix for the checkpoint filename
        """
        checkpoint_name = self.training_config["checkpoint_name"].replace(
            ".pt", f"_{suffix}.pt"
        )
        checkpoint_path = self.model_save_path / checkpoint_name

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": {
                    "input_size": len(self.features),
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "bidirectional": self.bidirectional,
                    "features": self.features,
                    "sequence_length": self.sequence_length,
                },
            },
            checkpoint_path,
        )
        print(f"\nModel saved to: {checkpoint_path}")

    def load_model(self, checkpoint_path: str) -> None:
        """
        Load a saved model checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Initialize model with saved configuration
        config = checkpoint["config"]
        self.model = RallyStateLSTM(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            bidirectional=config["bidirectional"],
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Model loaded from: {checkpoint_path}")

    def run(self) -> None:
        """Run the complete training pipeline."""
        # Load data
        self.load_data()

        # Train model
        self.train()

        # Evaluate on test set
        self.evaluate()


def main():
    """Main entry point for training."""
    trainer = RallyStateTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
