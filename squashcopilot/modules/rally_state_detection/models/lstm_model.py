"""
LSTM Model for Rally State Detection

This module implements a PyTorch LSTM model for binary classification
of rally states (active vs inactive).
"""

import torch
import torch.nn as nn


class RallyStateLSTM(nn.Module):
    """
    Bidirectional LSTM model for rally state detection.

    The model takes sequences of features (ball position, player positions)
    and predicts rally state (active=1, inactive=0) for each frame in the sequence.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        """
        Initialize the LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM layers
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability for regularization
            bidirectional: Whether to use bidirectional LSTM
        """
        super(RallyStateLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Calculate the size of LSTM output
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, 1),
            nn.Sigmoid(),  # Output probability for binary classification
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, sequence_length, 1)
            containing probabilities for each frame
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, sequence_length, hidden_size * num_directions)
        lstm_out, _ = self.lstm(x)

        # Apply fully connected layers to each timestep
        # out shape: (batch_size, sequence_length, 1)
        out = self.fc(lstm_out)

        return out

    def predict(self, x, threshold: float = 0.5):
        """
        Make binary predictions.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            threshold: Classification threshold (default: 0.5)

        Returns:
            Binary predictions (0 or 1) of shape (batch_size, sequence_length, 1)
        """
        with torch.no_grad():
            probabilities = self.forward(x)
            predictions = (probabilities > threshold).float()
        return predictions

    def get_model_info(self):
        """
        Get model architecture information.

        Returns:
            Dictionary containing model configuration
        """
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
