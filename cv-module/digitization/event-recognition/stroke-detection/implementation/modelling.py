import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle

# Assuming df_expanded is already defined
lstm_df = pd.read_csv("annotated_jsons/combined_data_2.csv")

# ----------------------------
# Step 1: Normalize coordinates
# ----------------------------
coord_cols = [
    col for col in lstm_df.columns if col.startswith("x") or col.startswith("y")
]
scaler = MinMaxScaler()
lstm_df[coord_cols] = scaler.fit_transform(lstm_df[coord_cols])
lstm_df.drop(columns=["player_id", "time", "index"], inplace=True)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Step 2: Label encoding
# ----------------------------
label_map = {"forehand": 0, "backhand": 1, "neither": 2}
lstm_df = lstm_df[lstm_df["event"].isin(label_map.keys())].copy()
lstm_df["event"] = lstm_df["event"].map(label_map).astype(int)

# Data preparation with non-overlapping windows
X = []
y = []
window_size = 16

# Process the data in non-overlapping chunks of size 16
for i in range(0, len(lstm_df), window_size):
    # Check if we have a full window
    if i + window_size <= len(lstm_df):
        window = lstm_df.iloc[i : i + window_size]
        # Only use the window if all rows have the same event
        if window["event"].nunique() == 1:
            X.append(
                window[coord_cols].values
            )  # Shape: (16, 24) - 16 timesteps, 24 features
            # Store the single event for this window
            y.append(window["event"].iloc[0])

X = np.array(X)  # shape should be (num_non_overlapping_windows, 16, 24)
y = np.array(y)  # shape should be (num_non_overlapping_windows,)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Label distribution: {np.unique(y, return_counts=True)}")

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size)
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]  # shape: (batch_size, hidden_size)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output


# Initialize model parameters
input_size = len(coord_cols)  # Number of features (24)
hidden_size = 4  # Same as the TF model
num_classes = 3  # Number of classes
num_epochs = 50

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Track metrics for plotting
train_acc_history = []
val_acc_history = []
best_val_acc = 0
patience = 3
counter = 0
best_model_state = None

# Training loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = correct / total
    train_acc_history.append(train_accuracy)

    # Validation phase
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    val_acc_history.append(val_accuracy)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}"
    )

    # Early stopping
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        best_model_state = model.state_dict()
        counter = 0
    # else:
    #     counter += 1
    #     if counter >= patience:
    #         print(f"Early stopping at epoch {epoch+1}")
    #         break

# Load the best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Evaluate the model
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print classification report
print(classification_report(all_labels, all_preds))

# Plot accuracy history
plt.figure()
plt.plot(range(1, len(train_acc_history) + 1), train_acc_history, "r--")
plt.plot(range(1, len(val_acc_history) + 1), val_acc_history, "b-")
plt.legend(["LSTM Training Accuracy", "LSTM Val Accuracy"])
plt.xlabel("Epoch")
plt.ylabel("LSTM Accuracy")
plt.show()

# Save the model
torch.save(model.state_dict(), "lstm_model.pt")

# To load the model later:
# model = LSTMModel(input_size, hidden_size, num_classes)
# model.load_state_dict(torch.load('lstm_model.pth'))
# model.eval()
