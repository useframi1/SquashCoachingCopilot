"""
LSTM Stroke Detection Training Script

This script trains an LSTM model for stroke classification (forehand vs backhand)
using pose keypoints from annotated CSV files.

Features:
- Loads annotated CSV files (4 for training, 1 for testing)
- Normalizes keypoints relative to player's hip center and torso length
- Creates 31-frame sequences for LSTM input
- Trains PyTorch LSTM classifier
- Evaluates and saves the best model
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from common.constants import KEYPOINT_NAMES, BODY_KEYPOINT_INDICES
from common.types.enums import StrokeType

# Note: We use normalize_keypoints_dataframe() instead of importing utils.normalize_keypoints()
# because the utils version operates on dict format while we need to process entire DataFrames.
# Both functions use the same normalization logic (hip center + torso length).


# ======================== Configuration ========================


class Config:
    """Training configuration"""

    # Paths
    ANNOTATION_DIR = Path("stroke_annotations")
    MODEL_DIR = Path("model/weights")
    CONFIG_PATH = Path("../../configs/stroke_detection.yaml")

    # Data split
    TEST_VIDEO = "video-2_strokes_annotated.csv"
    TRAIN_VIDEOS = [
        "video-1_strokes_annotated.csv",
        "video-3_strokes_annotated.csv",
        "video-4_strokes_annotated.csv",
        "video-5_strokes_annotated.csv",
    ]

    # Model hyperparameters
    SEQUENCE_LENGTH = (
        31  # Number of frames per sequence (using 31 instead of config's 15)
    )
    NUM_FEATURES = 24  # 12 keypoints × 2 coordinates (x, y)
    HIDDEN_SIZE = 128  # LSTM hidden units
    NUM_LAYERS = 2  # LSTM layers
    DROPOUT = 0.3  # Dropout rate
    NUM_CLASSES = 2  # forehand, backhand (excluding "neither" for training)

    # Training hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    PATIENCE = 10  # Early stopping patience

    # Use keypoint names from common constants
    KEYPOINT_NAMES = KEYPOINT_NAMES
    BODY_KEYPOINT_INDICES = BODY_KEYPOINT_INDICES

    # Normalization settings (from config)
    MIN_TORSO_LENGTH = 1e-6

    # Random seed for reproducibility
    RANDOM_SEED = 42

    @classmethod
    def load_from_yaml(cls, config_path: Path = None):
        """Load configuration from YAML file and merge with defaults."""
        if config_path is None:
            config_path = Path(__file__).parent / cls.CONFIG_PATH

        if config_path.exists():
            with open(config_path, "r") as f:
                yaml_config = yaml.safe_load(f)

            # Update normalization settings from config
            if "normalization" in yaml_config:
                cls.MIN_TORSO_LENGTH = yaml_config["normalization"].get(
                    "min_torso_length", cls.MIN_TORSO_LENGTH
                )

            print(f"✓ Loaded config from {config_path}")
        else:
            print(f"⚠ Config file not found at {config_path}, using defaults")


# ======================== Data Loading ========================


def load_csv_file(file_path: Path) -> pd.DataFrame:
    """
    Load a single CSV annotation file.

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame with stroke annotations
    """
    df = pd.read_csv(file_path)
    print(
        f"✓ Loaded {file_path.name}: {len(df)} frames, "
        f"{df['stroke_type'].nunique()} stroke types"
    )
    return df


def load_all_csvs(video_files: list, annotation_dir: Path) -> pd.DataFrame:
    """
    Load multiple CSV files and concatenate them.

    Args:
        video_files: List of CSV file names
        annotation_dir: Directory containing CSV files

    Returns:
        Combined DataFrame
    """
    dfs = []
    for video_file in video_files:
        file_path = annotation_dir / video_file
        if file_path.exists():
            df = load_csv_file(file_path)
            df["video_source"] = video_file
            dfs.append(df)
        else:
            print(f"⚠ Warning: {file_path} not found!")

    if not dfs:
        raise ValueError("No CSV files loaded!")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\n✓ Combined dataset: {len(combined_df)} total frames")
    print(f"Stroke type distribution:\n{combined_df['stroke_type'].value_counts()}")

    return combined_df


# ======================== Normalization ========================


def normalize_keypoints_dataframe(
    df: pd.DataFrame, keypoint_names: list, min_torso_length: float = 1e-6
) -> pd.DataFrame:
    """
    Normalize keypoints in a DataFrame relative to hip center and torso length.

    This uses the same normalization method as utils.normalize_keypoints()
    but operates on a DataFrame with multiple rows.

    Normalization makes the representation invariant to:
    - Player position in the frame (translation)
    - Player size/distance from camera (scale)

    Method:
    1. Hip center = average of left and right hip positions
    2. Shoulder center = average of left and right shoulder positions
    3. Torso length = distance between shoulder center and hip center
    4. Normalized_kp = (kp - hip_center) / torso_length

    Args:
        df: DataFrame with keypoint columns (kp_{name}_x, kp_{name}_y)
        keypoint_names: List of keypoint names
        min_torso_length: Minimum torso length to avoid division by zero

    Returns:
        DataFrame with added normalized keypoint columns (norm_{name}_x, norm_{name}_y)
    """
    df = df.copy()

    # Calculate hip center (average of left and right hip)
    hip_center_x = (df["kp_left_hip_x"] + df["kp_right_hip_x"]) / 2
    hip_center_y = (df["kp_left_hip_y"] + df["kp_right_hip_y"]) / 2

    # Calculate shoulder center (average of left and right shoulder)
    shoulder_center_x = (df["kp_left_shoulder_x"] + df["kp_right_shoulder_x"]) / 2
    shoulder_center_y = (df["kp_left_shoulder_y"] + df["kp_right_shoulder_y"]) / 2

    # Calculate torso length (Euclidean distance between centers)
    torso_length = np.sqrt(
        (shoulder_center_x - hip_center_x) ** 2
        + (shoulder_center_y - hip_center_y) ** 2
    )

    # Prevent division by zero (same logic as utils.normalize_keypoints)
    torso_length = np.where(torso_length < min_torso_length, 1.0, torso_length)

    # Normalize all keypoints
    for kp_name in keypoint_names:
        # Normalize x coordinate
        df[f"norm_{kp_name}_x"] = (df[f"kp_{kp_name}_x"] - hip_center_x) / torso_length
        # Normalize y coordinate
        df[f"norm_{kp_name}_y"] = (df[f"kp_{kp_name}_y"] - hip_center_y) / torso_length

    return df


# ======================== Sequence Creation ========================


def create_sequences(
    df: pd.DataFrame, keypoint_names: list, sequence_length: int = 31
) -> tuple:
    """
    Create sequences of consecutive frames for LSTM training.

    Each sequence consists of `sequence_length` consecutive frames from the same
    stroke event (same hit_frame and player_id), all with the same stroke_type label.

    Uses sliding window approach to create multiple sequences from each stroke event.

    Args:
        df: DataFrame with normalized keypoints
        keypoint_names: List of keypoint names
        sequence_length: Number of frames per sequence

    Returns:
        sequences: np.array of shape (N, sequence_length, num_features)
        labels: np.array of shape (N,) with stroke type labels
    """
    sequences = []
    labels = []

    # Build list of feature column names (normalized keypoints)
    feature_cols = []
    for kp_name in keypoint_names:
        feature_cols.append(f"norm_{kp_name}_x")
        feature_cols.append(f"norm_{kp_name}_y")

    # Group by stroke event (hit_frame and player_id)
    grouped = df.groupby(["hit_frame", "player_id"])

    valid_events = 0
    total_events = len(grouped)

    for (hit_frame, player_id), group in grouped:
        # Sort frames in temporal order
        group = group.sort_values("frame")

        # Only process if we have enough frames for at least one sequence
        if len(group) >= sequence_length:
            valid_events += 1

            # Get stroke type (should be consistent across all frames in group)
            stroke_type = group["stroke_type"].iloc[0]

            # Extract feature matrix (num_frames, num_features)
            features = group[feature_cols].values

            # Create sliding window sequences
            for i in range(len(features) - sequence_length + 1):
                sequence = features[i : i + sequence_length]
                sequences.append(sequence)
                labels.append(stroke_type)

    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels)

    print(
        f"✓ Created {len(sequences)} sequences from {valid_events}/{total_events} stroke events"
    )
    print(f"  Sequence shape: {sequences.shape}")
    print(f"  Label distribution: {np.unique(labels, return_counts=True)}")

    return sequences, labels


# ======================== PyTorch Dataset ========================


class StrokeSequenceDataset(Dataset):
    """
    PyTorch Dataset for stroke sequences.

    Each item is a tuple of:
    - sequence: Tensor of shape (sequence_length, num_features)
    - label: Integer label (encoded stroke type)
    """

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Args:
            sequences: np.array of shape (N, sequence_length, num_features)
            labels: np.array of shape (N,) with encoded labels
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple:
        return self.sequences[idx], self.labels[idx]


# ======================== LSTM Model ========================


class LSTMStrokeClassifier(nn.Module):
    """
    LSTM-based stroke classifier.

    Architecture:
    1. Multi-layer LSTM with dropout
    2. Take output from last time step
    3. Dropout layer
    4. Fully connected layer for classification

    Input: (batch_size, sequence_length, num_features)
    Output: (batch_size, num_classes)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_size: Number of features per time step
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(LSTMStrokeClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer with batch_first=True for (batch, seq, feature) input
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout if num_layers > 1 else 0
            ),  # Only use dropout if multiple layers
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM forward pass
        # out: (batch_size, sequence_length, hidden_size)
        # h_n, c_n: (num_layers, batch_size, hidden_size)
        out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # Take the output from the last time step
        # out[:, -1, :]: (batch_size, hidden_size)
        out = out[:, -1, :]

        # Apply dropout
        out = self.dropout(out)

        # Fully connected layer
        out = self.fc(out)

        return out


# ======================== Training Functions ========================


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple:
    """
    Train for one epoch.

    Returns:
        avg_loss: Average loss over all batches
        accuracy: Training accuracy (%)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for sequences, labels in tqdm(train_loader, desc="Training", leave=False):
        # Move to device
        sequences = sequences.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def validate_epoch(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple:
    """
    Validate for one epoch.

    Returns:
        avg_loss: Average loss over all batches
        accuracy: Validation accuracy (%)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, labels in tqdm(val_loader, desc="Validating", leave=False):
            # Move to device
            sequences = sequences.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


# ======================== Prediction Functions ========================


def predict(model: nn.Module, data_loader: DataLoader, device: torch.device) -> tuple:
    """
    Generate predictions for a dataset.

    Returns:
        all_preds: Predicted labels (encoded)
        all_labels: True labels (encoded)
        all_probs: Prediction probabilities
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for sequences, labels in tqdm(data_loader, desc="Predicting", leave=False):
            sequences = sequences.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(sequences)

            # Get probabilities using softmax
            probs = torch.softmax(outputs, dim=1)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def predict_single_sequence(
    model: nn.Module,
    sequence: np.ndarray,
    label_encoder: LabelEncoder,
    device: torch.device,
) -> tuple:
    """
    Predict stroke type for a single sequence.

    Args:
        model: Trained LSTM model
        sequence: np.array of shape (sequence_length, num_features)
        label_encoder: LabelEncoder for decoding predictions
        device: torch device

    Returns:
        predicted_label: String label (e.g., 'forehand', 'backhand')
        confidence: Confidence score (0-1)
        all_probs: Probabilities for all classes
    """
    model.eval()

    with torch.no_grad():
        # Add batch dimension and convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)

        # Forward pass
        output = model(sequence_tensor)

        # Get probabilities
        probs = torch.softmax(output, dim=1)

        # Get prediction and confidence
        confidence, predicted = torch.max(probs, 1)

        # Decode label
        predicted_label = label_encoder.inverse_transform([predicted.item()])[0]
        confidence_score = confidence.item()
        all_probs = probs.cpu().numpy()[0]

    return predicted_label, confidence_score, all_probs


# ======================== Visualization ========================


def plot_training_history(history: dict, save_path: Path):
    """Plot training and validation loss/accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", linewidth=2)
    ax1.plot(
        epochs, history["val_loss"], label="Validation Loss", marker="s", linewidth=2
    )
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(
        epochs,
        history["train_acc"],
        label="Train Accuracy",
        marker="o",
        linewidth=2,
        color="#2ecc71",
    )
    ax2.plot(
        epochs,
        history["val_acc"],
        label="Validation Accuracy",
        marker="s",
        linewidth=2,
        color="#e74c3c",
    )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Training vs Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])  # Set y-axis from 0 to 105% for better visualization

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Training history plot saved to {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list, save_path: Path
):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Confusion matrix saved to {save_path}")


# ======================== Main Training Loop ========================


def main(config: Config):
    """Main training function."""

    print("=" * 70)
    print("LSTM STROKE DETECTION TRAINING")
    print("=" * 70)

    # Load config from YAML
    config.load_from_yaml()

    # Set random seeds
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # Create output directory
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")

    # ==================== Load Data ====================
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    print("\nLoading training data...")
    train_df = load_all_csvs(config.TRAIN_VIDEOS, config.ANNOTATION_DIR)

    print("\nLoading test data...")
    test_df = load_csv_file(config.ANNOTATION_DIR / config.TEST_VIDEO)
    test_df["video_source"] = config.TEST_VIDEO

    # ==================== Normalize Keypoints ====================
    print("\n" + "=" * 70)
    print("NORMALIZING KEYPOINTS")
    print("=" * 70)

    print("\nNormalizing training data...")
    train_df = normalize_keypoints_dataframe(
        train_df, config.KEYPOINT_NAMES, config.MIN_TORSO_LENGTH
    )

    print("Normalizing test data...")
    test_df = normalize_keypoints_dataframe(
        test_df, config.KEYPOINT_NAMES, config.MIN_TORSO_LENGTH
    )

    print("✓ Normalization complete!")

    # ==================== Create Sequences ====================
    print("\n" + "=" * 70)
    print("CREATING SEQUENCES")
    print("=" * 70)

    print(f"\nCreating training sequences (length={config.SEQUENCE_LENGTH})...")
    X_train, y_train = create_sequences(
        train_df, config.KEYPOINT_NAMES, config.SEQUENCE_LENGTH
    )

    print(f"\nCreating test sequences (length={config.SEQUENCE_LENGTH})...")
    X_test, y_test = create_sequences(
        test_df, config.KEYPOINT_NAMES, config.SEQUENCE_LENGTH
    )

    print(f"\n✓ Training set: {X_train.shape[0]} sequences")
    print(f"✓ Test set: {X_test.shape[0]} sequences")

    # ==================== Encode Labels ====================
    print("\n" + "=" * 70)
    print("ENCODING LABELS")
    print("=" * 70)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print(f"\nLabel encoding:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {i}: {label}")

    print(f"\nTraining label distribution:")
    unique, counts = np.unique(y_train_encoded, return_counts=True)
    for label_idx, count in zip(unique, counts):
        print(
            f"  {label_encoder.classes_[label_idx]}: {count} "
            f"({count/len(y_train_encoded)*100:.1f}%)"
        )

    # ==================== Create Datasets ====================
    print("\n" + "=" * 70)
    print("CREATING DATASETS")
    print("=" * 70)

    train_dataset = StrokeSequenceDataset(X_train, y_train_encoded)
    test_dataset = StrokeSequenceDataset(X_test, y_test_encoded)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"\n✓ Train batches: {len(train_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")

    # ==================== Initialize Model ====================
    print("\n" + "=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)

    model = LSTMStrokeClassifier(
        input_size=config.NUM_FEATURES,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
    ).to(device)

    print(f"\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")

    # ==================== Training Setup ====================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # ==================== Training Loop ====================
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Print progress
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "label_encoder": label_encoder,
                "label_classes": label_encoder.classes_.tolist(),  # For easier access
                "config": {
                    "input_size": config.NUM_FEATURES,
                    "hidden_size": config.HIDDEN_SIZE,
                    "num_layers": config.NUM_LAYERS,
                    "num_classes": config.NUM_CLASSES,
                    "dropout": config.DROPOUT,
                    "sequence_length": config.SEQUENCE_LENGTH,
                },
            }
            torch.save(checkpoint, config.MODEL_DIR / "lstm_model.pt")
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  Early stopping: {patience_counter}/{config.PATIENCE}")

        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
            break

    print("\n✓ Training completed!")

    # ==================== Plot Training History ====================
    plot_training_history(history, config.MODEL_DIR / "training_history.png")

    # ==================== Evaluation ====================
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    # Load best model
    checkpoint = torch.load(config.MODEL_DIR / "lstm_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\n✓ Loaded best model from epoch {checkpoint['epoch']}")

    # Generate predictions
    print("\nGenerating predictions on test set...")
    y_pred, y_true, y_probs = predict(model, test_loader, device)

    # Convert encoded labels back to original labels
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_true_labels = label_encoder.inverse_transform(y_true)

    # Print classification report
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(
        classification_report(
            y_true_labels, y_pred_labels, target_names=label_encoder.classes_, digits=4
        )
    )

    print("################LABEL CLASSES################")
    print("!!!!!!!!!!!!!!!!!!!! TRUE LABELS !!!!!!!!!!!!!!!!!!!!")
    print(y_true_labels)
    print("!!!!!!!!!!!!!!!!!!!! PREDICTED LABELS !!!!!!!!!!!!!!!!!!!!")
    print(y_pred_labels)

    # Overall accuracy
    accuracy = accuracy_score(y_true_labels, y_pred_labels) * 100
    print(f"\n✓ Overall Test Accuracy: {accuracy:.2f}%")

    # Plot confusion matrix
    plot_confusion_matrix(
        y_true_labels,
        y_pred_labels,
        label_encoder.classes_.tolist(),
        config.MODEL_DIR / "confusion_matrix.png",
    )

    # ==================== Summary ====================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nModel Architecture:")
    print(f"  - Input size: {config.NUM_FEATURES} features")
    print(f"  - Sequence length: {config.SEQUENCE_LENGTH} frames")
    print(f"  - LSTM hidden size: {config.HIDDEN_SIZE}")
    print(f"  - LSTM layers: {config.NUM_LAYERS}")
    print(f"  - Dropout: {config.DROPOUT}")
    print(f"  - Output classes: {config.NUM_CLASSES}")

    print(f"\nTraining:")
    print(f"  - Training sequences: {len(X_train)}")
    print(f"  - Test sequences: {len(X_test)}")
    print(f"  - Epochs trained: {len(history['train_loss'])}")

    print(f"\nPerformance:")
    print(f"  - Best validation loss: {checkpoint['val_loss']:.4f}")
    print(f"  - Test accuracy: {accuracy:.2f}%")

    print(f"\nSaved Files:")
    print(f"  - Model: {config.MODEL_DIR / 'lstm_model.pt'}")
    print(f"  - Training history: {config.MODEL_DIR / 'training_history.png'}")
    print(f"  - Confusion matrix: {config.MODEL_DIR / 'confusion_matrix.png'}")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM stroke classifier")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=31,
        help="Number of frames per sequence (default: 31)",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=128, help="LSTM hidden size (default: 128)"
    )
    parser.add_argument(
        "--num-layers", type=int, default=2, help="Number of LSTM layers (default: 2)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs (default: 50)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )

    args = parser.parse_args()

    # Update config with command line arguments
    config = Config()
    config.SEQUENCE_LENGTH = args.sequence_length
    config.HIDDEN_SIZE = args.hidden_size
    config.NUM_LAYERS = args.num_layers
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr

    # Run training
    main(config)
