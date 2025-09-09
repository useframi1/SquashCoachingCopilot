import torch
import torch.nn as nn
from torchvision import models


class SquashRallyDetector(nn.Module):
    def __init__(self, feature_dim, lstm_hidden_dim, dropout=0.5):
        super(SquashRallyDetector, self).__init__()

        # Load pre-trained ResNet as feature extractor
        self.cnn = models.resnet50(weights="IMAGENET1K_V2")
        # Remove the final FC layer
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

        # Freeze early CNN layers
        for name, param in self.cnn.named_parameters():
            if "layer4" not in name:  # Only fine-tune the last layer
                param.requires_grad = False

        # Add dimension reduction layer
        self.dim_reduction = nn.Linear(2048, feature_dim)

        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
            bidirectional=True,
        )

        # Final classification layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 64),  # * 2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # x shape: [batch_size, sequence_length, channels, height, width]
        batch_size, sequence_length, c, h, w = x.shape

        # Reshape for CNN processing
        x = x.view(batch_size * sequence_length, c, h, w)

        # Extract features with CNN
        with torch.no_grad():  # Use no_grad for frozen layers to save memory
            features = self.cnn(x)

        # Flatten features
        features = features.view(batch_size * sequence_length, -1)

        # Apply dimension reduction
        features = self.dim_reduction(features)

        # Reshape features for LSTM
        features = features.view(batch_size, sequence_length, -1)

        # Process with LSTM
        lstm_out, _ = self.lstm(features)

        # Use last time step output for classification
        lstm_out = lstm_out[:, -1, :]

        # Final classification
        output = self.fc(lstm_out)

        return output
