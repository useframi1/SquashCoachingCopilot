import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime


class TrainingPipeline:
    """Pipeline for training, evaluating and using squash rally detection models."""

    def __init__(self, config, squash_rally_dataset_class, squash_rally_detector_class):
        """
        Initialize the pipeline with configuration and required classes.

        Args:
            config (dict): Configuration dictionary with all parameters
            squash_rally_dataset_class: The dataset class to use
            squash_rally_detector_class: The model class to use
        """
        self.config = config
        self.SquashRallyDataset = squash_rally_dataset_class
        self.SquashRallyDetector = squash_rally_detector_class

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create directories if they don't exist
        os.makedirs(self.config["model_save_path"], exist_ok=True)

        # Initialize tracking variables
        self.train_losses = []
        self.val_losses = []
        self.train_f1s = []
        self.val_f1s = []
        self.best_val_f1 = 0
        self.best_model_path = None

        # Initialize other properties that will be set during pipeline execution
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def load_annotations(self):
        """Load annotations from JSON file"""
        with open(self.config["annotation_file"], "r") as f:
            return json.load(f)

    def get_video_paths(self):
        """Get paths to all video files in data directory"""
        video_extensions = [".mp4", ".avi", ".mov"]
        video_paths = []

        for root, _, files in os.walk(self.config["data_path"]):
            for file in files:
                if any(file.endswith(ext) for ext in video_extensions):
                    video_paths.append(os.path.join(root, file))

        return video_paths

    def prepare_datasets(self, video_paths, annotations):
        """
        Prepare train, validation and test datasets.

        Args:
            video_paths (list): List of paths to video files
            annotations (dict): Dictionary of video annotations

        Returns:
            tuple: Train, validation and test datasets
        """
        # Define transform
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    (self.config["image_size"], self.config["image_size"])
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Split videos into train, validation, test sets
        video_ids = list(annotations.keys())
        train_ids, temp_ids = train_test_split(
            video_ids, test_size=0.3, random_state=42
        )
        print(f"Train: {len(train_ids)}, Temp: {len(temp_ids)}")
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

        # Create annotation subsets
        train_annotations = {k: annotations[k] for k in train_ids}
        val_annotations = {k: annotations[k] for k in val_ids}
        test_annotations = {k: annotations[k] for k in test_ids}

        # Create datasets
        train_dataset = self.SquashRallyDataset(
            video_paths, train_annotations, self.config["sequence_length"], transform
        )
        val_dataset = self.SquashRallyDataset(
            video_paths, val_annotations, self.config["sequence_length"], transform
        )
        test_dataset = self.SquashRallyDataset(
            video_paths, test_annotations, self.config["sequence_length"], transform
        )

        print(
            f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset

    def setup_model(self):
        """Initialize model, loss function, optimizer and scheduler"""
        # Initialize model
        self.model = self.SquashRallyDetector(
            feature_dim=self.config["feature_dim"],
            lstm_hidden_dim=self.config["lstm_hidden_dim"],
            dropout=self.config["dropout"],
        ).to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config["learning_rate"],
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def train_epoch(self, dataloader):
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader for the training set

        Returns:
            tuple: Loss, precision, recall, F1 score
        """
        self.model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []

        for sequences, labels in tqdm(dataloader):
            # Move data to device
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs.squeeze(), labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            # Store predictions and labels for metrics
            preds = (outputs.squeeze() > 0.5).float().cpu().detach().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        return epoch_loss / len(dataloader), precision, recall, f1

    def evaluate(self, dataloader):
        """
        Evaluate model on validation or test set.

        Args:
            dataloader: DataLoader for evaluation

        Returns:
            tuple: Loss, precision, recall, F1 score
        """
        self.model.eval()
        eval_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in tqdm(dataloader):
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(sequences)
                loss = self.criterion(outputs.squeeze(), labels)

                eval_loss += loss.item()

                preds = (outputs.squeeze() > 0.5).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        return eval_loss / len(dataloader), precision, recall, f1

    def save_model(self, epoch, val_f1):
        """
        Save the current model.

        Args:
            epoch (int): Current epoch
            val_f1 (float): Validation F1 score

        Returns:
            str: Path to the saved model
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(
            self.config["model_save_path"],
            f"model_epoch_{epoch+1}_f1_{val_f1:.4f}_{timestamp}.pt",
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_f1": val_f1,
                "config": self.config,
            },
            model_path,
        )

        print(f"Saved new best model with F1: {val_f1:.4f}")
        return model_path

    def plot_training_history(self):
        """Plot and save training history graphs"""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.train_f1s, label="Train F1")
        plt.plot(self.val_f1s, label="Validation F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title("Training and Validation F1 Score")
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config["model_save_path"], "training_history.png")
        )
        plt.close()

    def train(self):
        """
        Execute the complete training pipeline.

        Returns:
            str: Path to the best model
        """
        # Load data
        annotations = self.load_annotations()
        video_paths = self.get_video_paths()

        # Prepare datasets
        train_dataset, val_dataset, test_dataset = self.prepare_datasets(
            video_paths, annotations
        )

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Setup model and training components
        self.setup_model()

        # Training loop
        patience_counter = 0

        for epoch in range(self.config["num_epochs"]):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")

            # Train
            train_loss, train_precision, train_recall, train_f1 = self.train_epoch(
                self.train_loader
            )

            # Validate
            val_loss, val_precision, val_recall, val_f1 = self.evaluate(self.val_loader)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Store metrics for plotting
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_f1s.append(train_f1)
            self.val_f1s.append(val_f1)

            # Print progress
            print(
                f"Train - Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}"
            )
            print(
                f"Val   - Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}"
            )

            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                patience_counter = 0

                # Remove previous best model if it exists
                if self.best_model_path and os.path.exists(self.best_model_path):
                    os.remove(self.best_model_path)

                # Save new best model
                self.best_model_path = self.save_model(epoch, val_f1)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Plot training history
        self.plot_training_history()

        # Test the best model
        if self.best_model_path:
            self.test_best_model()

        return self.best_model_path

    def test_best_model(self):
        """Evaluate the best model on the test set"""
        if not self.best_model_path or not os.path.exists(self.best_model_path):
            print("No best model found to test")
            return

        # Load best model
        checkpoint = torch.load(self.best_model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Evaluate on test set
        test_loss, test_precision, test_recall, test_f1 = self.evaluate(
            self.test_loader
        )

        print("\nTest Set Evaluation:")
        print(
            f"Loss: {test_loss:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}"
        )

        # Save test results
        test_results = {
            "test_loss": test_loss,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "model_path": self.best_model_path,
        }

        with open(
            os.path.join(self.config["model_save_path"], "test_results.json"), "w"
        ) as f:
            json.dump(test_results, f, indent=4)

    def load_model(self, model_path):
        """
        Load a previously trained model.

        Args:
            model_path (str): Path to the model file

        Returns:
            The loaded model
        """
        checkpoint = torch.load(model_path, map_location=self.device)

        # Initialize model architecture
        model = self.SquashRallyDetector(
            feature_dim=self.config["feature_dim"],
            lstm_hidden_dim=self.config["lstm_hidden_dim"],
            dropout=self.config["dropout"],
        ).to(self.device)

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()  # Set to evaluation mode

        return model

    def predict(self, model, video_sequence):
        """
        Make predictions on a video sequence.

        Args:
            model: The loaded model to use for prediction
            video_sequence: Preprocessed video sequence tensor

        Returns:
            float: Probability that the sequence contains a squash rally
        """
        model.eval()
        with torch.no_grad():
            # Ensure sequence is on the correct device
            video_sequence = video_sequence.to(self.device)
            # Make prediction
            output = model(video_sequence)
            # Return probability
            return output.squeeze().item()
