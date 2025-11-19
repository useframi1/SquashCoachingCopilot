"""
Stroke Detection Model Training

This module provides functionality to train an LSTM model for stroke type detection
using annotated stroke data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from squashcopilot.common.utils import load_config
from squashcopilot.common.constants import KEYPOINT_NAMES
from squashcopilot.modules.stroke_detection.model.lstm_classifier import (
    LSTMStrokeClassifier,
)


class StrokeDataset(Dataset):
    """PyTorch Dataset for stroke detection sequences."""

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Initialize the dataset.

        Args:
            sequences: Array of shape (num_sequences, sequence_length, num_features)
            labels: Array of shape (num_sequences,) with class indices
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class StrokeTrainer:
    """Trainer class for LSTM-based stroke detection."""

    def __init__(self, config_name: str = "stroke_detection"):
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
        self.model_save_dir = project_root / self.training_config["model_save_dir"]
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        # Extract configuration
        self.keypoint_names = KEYPOINT_NAMES
        self.sequence_length = self.training_config["sequence_length"]
        self.train_test_split = self.training_config.get("train_test_split", 0.8)
        self.validation_split = self.training_config.get("validation_split", 0.1)
        self.batch_size = self.training_config["batch_size"]
        self.epochs = self.training_config["epochs"]
        self.learning_rate = self.training_config["learning_rate"]

        # Model configuration
        model_config = self.training_config["model"]
        self.hidden_size = model_config["hidden_size"]
        self.num_layers = model_config.get("num_layers", 1)
        self.dropout = model_config["dropout"]

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize model, criterion, optimizer (will be set during training)
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.label_encoder = LabelEncoder()

        # Data containers
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def _normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize keypoints relative to hip center and torso length.

        Args:
            keypoints: Array of shape (sequence_length, num_keypoints * 2)
                      where keypoints are in format [x1, y1, x2, y2, ...]

        Returns:
            Normalized keypoints array of same shape
        """
        # Keypoints order matches KEYPOINT_NAMES
        # Indices: left_shoulder(0,1), right_shoulder(2,3), ..., left_hip(12,13), right_hip(14,15), ...

        # Extract hip keypoints (indices 6 and 7 in KEYPOINT_NAMES)
        left_hip_idx = self.keypoint_names.index("left_hip")
        right_hip_idx = self.keypoint_names.index("right_hip")

        # Extract shoulder keypoints (indices 0 and 1 in KEYPOINT_NAMES)
        left_shoulder_idx = self.keypoint_names.index("left_shoulder")
        right_shoulder_idx = self.keypoint_names.index("right_shoulder")

        # Calculate hip center for each frame
        hip_center_x = (
            keypoints[:, left_hip_idx * 2] + keypoints[:, right_hip_idx * 2]
        ) / 2
        hip_center_y = (
            keypoints[:, left_hip_idx * 2 + 1] + keypoints[:, right_hip_idx * 2 + 1]
        ) / 2

        # Calculate shoulder center for each frame
        shoulder_center_x = (
            keypoints[:, left_shoulder_idx * 2] + keypoints[:, right_shoulder_idx * 2]
        ) / 2
        shoulder_center_y = (
            keypoints[:, left_shoulder_idx * 2 + 1]
            + keypoints[:, right_shoulder_idx * 2 + 1]
        ) / 2

        # Calculate torso length for each frame
        torso_length = np.sqrt(
            (shoulder_center_x - hip_center_x) ** 2
            + (shoulder_center_y - hip_center_y) ** 2
        )

        # Prevent division by zero
        torso_length = np.where(
            torso_length < self.config["normalization"]["min_torso_length"],
            1.0,
            torso_length,
        )

        # Normalize all keypoints
        normalized = keypoints.copy()
        for i in range(0, keypoints.shape[1], 2):  # Iterate over x coordinates
            normalized[:, i] = (keypoints[:, i] - hip_center_x) / torso_length  # x
            normalized[:, i + 1] = (
                keypoints[:, i + 1] - hip_center_y
            ) / torso_length  # y

        return normalized

    def load_data(self) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], List[str]]:
        """
        Load and prepare all stroke annotation data, organized by video.

        Returns:
            Tuple of (video_data, label_names)
            - video_data: Dict mapping video_name -> (sequences, labels)
                - sequences: Array of shape (num_sequences, sequence_length, num_features)
                - labels: Array of shape (num_sequences,) with class indices
            - label_names: List of class names
        """
        print("\n" + "=" * 70)
        print("LOADING STROKE DATA")
        print("=" * 70)

        video_data = {}
        all_labels_str = []

        # Find all video subdirectories in data directory
        video_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        if not video_dirs:
            raise FileNotFoundError(f"No video subdirectories found in {self.data_dir}")

        print(f"\nFound {len(video_dirs)} video directories:")
        for video_dir in video_dirs:
            print(f"  - {video_dir.name}")

        # Process each video directory
        for video_dir in video_dirs:
            video_name = video_dir.name

            # Look for the strokes annotated CSV file
            csv_file = video_dir / f"{video_name}_annotations.csv"

            if not csv_file.exists():
                print(f"\n⚠ Skipping {video_name}: No strokes annotated CSV found")
                continue

            print(f"\nProcessing {csv_file.name}...")
            df = pd.read_csv(csv_file)

            video_sequences = []
            video_labels = []

            # Group by hit_frame to get sequences
            grouped = df.groupby("hit_frame")

            for hit_frame, group_df in grouped:
                # Sort by frame to ensure temporal order
                group_df = group_df.sort_values("frame")

                # Check sequence length
                if len(group_df) != self.sequence_length:
                    continue  # Skip incomplete sequences

                # Extract keypoint features (all keypoints in order)
                features = []
                for _, row in group_df.iterrows():
                    kp_values = []
                    for kp_name in self.keypoint_names:
                        kp_values.append(row[f"kp_{kp_name}_x"])
                        kp_values.append(row[f"kp_{kp_name}_y"])
                    features.append(kp_values)

                sequence = np.array(features, dtype=np.float32)

                # Normalize keypoints
                sequence = self._normalize_keypoints(sequence)

                # Get stroke label (should be same for all frames in the group)
                stroke_type = group_df.iloc[0]["stroke_type"]

                video_sequences.append(sequence)
                video_labels.append(stroke_type)
                all_labels_str.append(stroke_type)

            video_data[video_name] = (
                np.array(video_sequences, dtype=np.float32),
                video_labels,
            )
            print(f"  ✓ Loaded {len(video_sequences)} sequences")

        # Encode labels (fit on all labels across all videos)
        all_labels_str = np.array(all_labels_str)
        self.label_encoder.fit(all_labels_str)
        label_names = self.label_encoder.classes_.tolist()

        # Encode labels for each video
        for video_name in video_data:
            sequences, labels_str = video_data[video_name]
            labels = self.label_encoder.transform(labels_str)
            video_data[video_name] = (sequences, labels)

        # Print summary
        total_sequences = sum(len(seqs) for seqs, _ in video_data.values())
        print(f"\n{'=' * 70}")
        print("DATA SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total videos: {len(video_data)}")
        print(f"Total sequences: {total_sequences}")
        if total_sequences > 0:
            first_seq = next(iter(video_data.values()))[0][0]
            print(f"Sequence shape: {first_seq.shape}")
        print(f"Classes: {label_names}")
        print(f"Class distribution:")
        for label_name in label_names:
            count = np.sum(all_labels_str == label_name)
            percentage = count / len(all_labels_str) * 100
            print(f"  {label_name}: {count} ({percentage:.1f}%)")

        return video_data, label_names

    def prepare_dataloaders(
        self, video_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """
        Split data by video and create DataLoaders.

        Args:
            video_data: Dict mapping video_name -> (sequences, labels)
        """
        print(f"\n{'=' * 70}")
        print("PREPARING DATALOADERS (VIDEO-LEVEL SPLIT)")
        print(f"{'=' * 70}")

        # Get list of video names and shuffle them
        video_names = list(video_data.keys())
        np.random.shuffle(video_names)

        # Calculate split sizes based on number of videos
        n_videos = len(video_names)
        n_test_videos = max(1, int(n_videos * (1 - self.train_test_split)))
        n_train_val_videos = n_videos - n_test_videos
        n_val_videos = max(1, int(n_train_val_videos * self.validation_split))
        n_train_videos = n_train_val_videos - n_val_videos

        # Split video names
        train_videos = video_names[:n_train_videos]
        val_videos = video_names[n_train_videos : n_train_videos + n_val_videos]
        test_videos = video_names[n_train_videos + n_val_videos :]

        print(f"\nVideo split:")
        print(f"  Train videos: {train_videos}")
        print(f"  Val videos:   {val_videos}")
        print(f"  Test videos:  {test_videos}")

        # Aggregate sequences for each split
        train_sequences, train_labels = [], []
        val_sequences, val_labels = [], []
        test_sequences, test_labels = [], []

        for video_name in train_videos:
            seqs, lbls = video_data[video_name]
            train_sequences.append(seqs)
            train_labels.append(lbls)

        for video_name in val_videos:
            seqs, lbls = video_data[video_name]
            val_sequences.append(seqs)
            val_labels.append(lbls)

        for video_name in test_videos:
            seqs, lbls = video_data[video_name]
            test_sequences.append(seqs)
            test_labels.append(lbls)

        # Concatenate all sequences
        train_sequences = np.concatenate(train_sequences, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        val_sequences = np.concatenate(val_sequences, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        test_sequences = np.concatenate(test_sequences, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        n_total = len(train_sequences) + len(val_sequences) + len(test_sequences)
        print(f"\nSequence split:")
        print(
            f"  Train: {len(train_sequences)} sequences ({len(train_sequences) / n_total * 100:.1f}%)"
        )
        print(
            f"  Val:   {len(val_sequences)} sequences ({len(val_sequences) / n_total * 100:.1f}%)"
        )
        print(
            f"  Test:  {len(test_sequences)} sequences ({len(test_sequences) / n_total * 100:.1f}%)"
        )

        # Create datasets
        train_dataset = StrokeDataset(train_sequences, train_labels)
        val_dataset = StrokeDataset(val_sequences, val_labels)
        test_dataset = StrokeDataset(test_sequences, test_labels)

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        print(f"\nDataLoaders created:")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches:   {len(self.val_loader)}")
        print(f"  Test batches:  {len(self.test_loader)}")

    def initialize_model(self, num_classes: int, input_size: int) -> None:
        """
        Initialize the LSTM model, criterion, and optimizer.

        Args:
            num_classes: Number of stroke classes
            input_size: Number of input features
        """
        self.model = LSTMStrokeClassifier(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=num_classes,
            dropout=self.dropout,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print(f"\n{'=' * 70}")
        print("MODEL INITIALIZED")
        print(f"{'=' * 70}")
        print(f"Architecture: LSTM")
        print(f"Input size: {input_size}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Num layers: {self.num_layers}")
        print(f"Num classes: {num_classes}")
        print(f"Dropout: {self.dropout}")
        print(f"Learning rate: {self.learning_rate}")

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for sequences, labels in tqdm(self.train_loader, desc="Training"):
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

            # Store predictions
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in self.val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def evaluate(self) -> dict:
        """Evaluate the model on test set."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in self.test_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(sequences)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        recall = recall_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": conf_matrix,
        }

    def train(self) -> None:
        """Main training loop."""
        print(f"\n{'=' * 70}")
        print("TRAINING")
        print(f"{'=' * 70}")

        best_val_accuracy = 0.0
        best_model_path = self.model_save_dir / "stroke_lstm_best.pt"

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print("-" * 70)

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                self.save_model(best_model_path)
                print(f"✓ New best model saved (val_acc: {val_acc:.4f})")

        print(f"\n{'=' * 70}")
        print("TRAINING COMPLETE")
        print(f"{'=' * 70}")
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")

    def save_model(self, path: Path) -> None:
        """Save model checkpoint."""
        # Get input size from first batch
        sample_batch = next(iter(self.train_loader))
        input_size = sample_batch[0].shape[-1]

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": {
                "input_size": input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_classes": len(self.label_encoder.classes_),
                "dropout": self.dropout,
                "sequence_length": self.sequence_length,
            },
            "label_encoder": self.label_encoder,
            "label_classes": self.label_encoder.classes_.tolist(),
        }

        torch.save(checkpoint, path)

    def run(self) -> None:
        """Run the complete training pipeline."""
        # Load data (organized by video)
        video_data, label_names = self.load_data()

        # Prepare dataloaders (split by video)
        self.prepare_dataloaders(video_data)

        # Initialize model
        # Get input size from first video's first sequence
        first_video_sequences = next(iter(video_data.values()))[0]
        input_size = first_video_sequences.shape[-1]  # Number of features
        num_classes = len(label_names)
        self.initialize_model(num_classes, input_size)

        # Train
        self.train()

        # Evaluate on test set
        print(f"\n{'=' * 70}")
        print("FINAL EVALUATION ON TEST SET")
        print(f"{'=' * 70}")

        metrics = self.evaluate()

        print(f"\nTest Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"Classes: {label_names}")
        print(metrics["confusion_matrix"])


if __name__ == "__main__":
    trainer = StrokeTrainer()
    trainer.run()
