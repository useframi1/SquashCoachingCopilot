import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ==================== Loading Dataset ====================
df_1 = pd.read_json("annotated_jsons/AO_Output.json")
df_2 = pd.read_json("annotated_jsons/FA Output.json")
df_3 = pd.read_json("annotated_jsons/video_4_annotated.json")
df_4 = pd.read_json("annotated_jsons/integrated_data.json")
df_5 = pd.read_json("annotated_jsons/nadines_video_annotated.json")
df_6 = pd.read_json("annotated_jsons/video_1_annotated.json")

# Add index column to each dataframe
df_1["index"] = 1
df_2["index"] = 2
df_3["index"] = 3
df_4["index"] = 4
df_5["index"] = 5
df_6["index"] = 6

# Print event value counts
dfs = [df_1, df_2, df_3, df_4, df_5, df_6]
for idx, df in enumerate(dfs, 1):
    print(f"df_{idx} event value counts:")
    print(df["event"].value_counts())
    print("-" * 40)

# Create train and test sets
train_df = pd.concat([df_5, df_2, df_1, df_4, df_6], ignore_index=True)
test_df = df_3.copy()

print(f"Train shape: {train_df.shape}")
print(f"Train event counts:\n{train_df['event'].value_counts()}")
print(f"Test shape: {test_df.shape}")
print(f"Test event counts:\n{test_df['event'].value_counts()}")

# ==================== Expand Keypoints ====================
def expand_df(df):
    expanded_df = (
        df["keypoints"]
        .apply(
            lambda person: {f"x_{part}": person[part]["x"] for part in person}
            | {f"y_{part}": person[part]["y"] for part in person}
        )
        .apply(pd.Series)
    )
    df_expanded = pd.concat([df, expanded_df], axis=1)
    df_expanded.drop(columns=["keypoints"], inplace=True)
    return df_expanded

train_df_expanded = expand_df(train_df)
test_df_expanded = expand_df(test_df)

print(f"Expanded train shape: {train_df_expanded.shape}")

# ==================== Normalize Keypoints ====================
def normalize_keypoints_df(df):
    """
    Normalize all keypoints in the dataframe using body-relative normalization
    REPLACES original x_, y_ columns with normalized values
    """
    df_norm = df.copy()

    keypoint_names = [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    for idx, row in df.iterrows():
        # Calculate hip center
        hip_center_x = (row["x_left_hip"] + row["x_right_hip"]) / 2
        hip_center_y = (row["y_left_hip"] + row["y_right_hip"]) / 2

        # Calculate shoulder center
        shoulder_center_x = (row["x_left_shoulder"] + row["x_right_shoulder"]) / 2
        shoulder_center_y = (row["y_left_shoulder"] + row["y_right_shoulder"]) / 2

        # Calculate torso length
        torso_length = np.sqrt(
            (shoulder_center_x - hip_center_x) ** 2
            + (shoulder_center_y - hip_center_y) ** 2
        )

        if torso_length < 1e-6:
            torso_length = 1.0

        # REPLACE original columns with normalized values
        for name in keypoint_names:
            df_norm.at[idx, f"x_{name}"] = (
                row[f"x_{name}"] - hip_center_x
            ) / torso_length
            df_norm.at[idx, f"y_{name}"] = (
                row[f"y_{name}"] - hip_center_y
            ) / torso_length

    return df_norm

train_df_expanded = normalize_keypoints_df(train_df_expanded)
test_df_expanded = normalize_keypoints_df(test_df_expanded)

# ==================== Add Sequences ====================
def add_sequence(df_expanded):
    event_indices = df_expanded[df_expanded["event"].notnull()].index

    selected_indices = set()
    neither_count = 0
    MAX_NEITHER_SEGMENTS = 100

    # Step 1: Annotate labeled events with 7 frames before and 7 frames after
    for idx in event_indices:
        label = df_expanded.at[idx, "event"]
        group_id = df_expanded.at[idx, "index"]
        
        # Add the labeled frame itself
        selected_indices.add(idx)
        df_expanded.at[idx, "event"] = label

        # Add 7 frames BEFORE
        for offset in range(1, 8):
            prev_idx = idx - offset
            if prev_idx < 0:
                break

            if (
                pd.notnull(df_expanded.at[prev_idx, "event"])
                or df_expanded.at[prev_idx, "index"] != group_id
            ):
                break

            selected_indices.add(prev_idx)
            df_expanded.at[prev_idx, "event"] = label

        # Add 7 frames AFTER
        for offset in range(1, 8):
            next_idx = idx + offset
            if next_idx >= len(df_expanded):
                break

            if (
                pd.notnull(df_expanded.at[next_idx, "event"])
                or df_expanded.at[next_idx, "index"] != group_id
            ):
                break

            selected_indices.add(next_idx)
            df_expanded.at[next_idx, "event"] = label

    # Step 2: Annotate null event segments in 15-frame "neither" batches
    null_indices = df_expanded[df_expanded["event"].isnull()].index
    null_indices = sorted(null_indices)

    i = 0
    while i < len(null_indices):
        if neither_count >= MAX_NEITHER_SEGMENTS:
            break

        start_idx = null_indices[i]
        group_id = df_expanded.at[start_idx, "index"]

        # Collect consecutive nulls in same group
        segment = [start_idx]
        for j in range(i + 1, len(null_indices)):
            current_idx = null_indices[j]
            prev_idx = null_indices[j - 1]

            if (
                current_idx == prev_idx + 1
                and df_expanded.at[current_idx, "index"] == group_id
            ):
                segment.append(current_idx)
            else:
                break

        # Process in batches of 15
        for k in range(0, len(segment), 15):
            if neither_count >= MAX_NEITHER_SEGMENTS:
                break

            batch = segment[k : k + 15]
            if len(batch) == 15:
                for idx in batch:
                    selected_indices.add(idx)
                    df_expanded.at[idx, "event"] = "neither"
                neither_count += 1

        i += len(segment)

    # Step 3: Final filtering
    df_expanded = df_expanded.loc[sorted(selected_indices)].reset_index(drop=True)

    print(f"Number of 'neither' segments (15-frame batches): {neither_count}")

    return df_expanded

train_df_expanded = add_sequence(train_df_expanded)
test_df_expanded = add_sequence(test_df_expanded)

print(f"Train event counts after sequencing:\n{train_df_expanded['event'].value_counts()}")
print(f"Test event counts after sequencing:\n{test_df_expanded['event'].value_counts()}")

# Drop unnecessary columns
train_df_expanded.drop(columns=["player_id", "time", "index"], inplace=True)
test_df_expanded.drop(columns=["player_id", "time", "index"], inplace=True)

# ==================== Label Encoding ====================
label_map = {"forehand": 0, "backhand": 1, "neither": 2}
train_df_expanded = train_df_expanded[
    train_df_expanded["event"].isin(label_map.keys())
].copy()
train_df_expanded["event"] = train_df_expanded["event"].map(label_map).astype(int)

test_df_expanded = test_df_expanded[
    test_df_expanded["event"].isin(label_map.keys())
].copy()
test_df_expanded["event"] = test_df_expanded["event"].map(label_map).astype(int)

# Get coordinate columns
coord_cols = [
    col
    for col in train_df_expanded.columns
    if col.startswith("x") or col.startswith("y")
]

# ==================== Create Sequences ====================
def create_sequences(df_expanded, coord_cols, window_size=15):
    X = []
    y = []
    
    for i in range(0, len(df_expanded), window_size):
        if i + window_size <= len(df_expanded):
            window = df_expanded.iloc[i : i + window_size]
            
            if window["event"].nunique() == 1:
                X.append(window[coord_cols].values)
                y.append(window["event"].iloc[0])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

window_size = 15
X_train, y_train = create_sequences(train_df_expanded, coord_cols, window_size)
X_test, y_test = create_sequences(test_df_expanded, coord_cols, window_size)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"Label distribution: {np.unique(y_train, return_counts=True)}")

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Label distribution: {np.unique(y_test, return_counts=True)}")

# ==================== PyTorch Dataset ====================
class SquashDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==================== LSTM Model ====================
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.1):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden = hidden[-1]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out

# ==================== Training Setup ====================
# Create datasets and dataloaders
train_dataset = SquashDataset(X_train, y_train)
test_dataset = SquashDataset(X_test, y_test)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
input_size = len(coord_cols)
hidden_size = 16
num_classes = 3
dropout = 0.1

model = LSTMClassifier(input_size, hidden_size, num_classes, dropout).to(device)

# Compute class weights
class_weights = compute_class_weight('balanced', 
                                    classes=np.unique(y_train), 
                                    y=y_train)
class_weights = torch.FloatTensor(class_weights).to(device)
print(f"Class weights: {dict(enumerate(class_weights.cpu().numpy()))}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters())

# ==================== Training Loop ====================
epochs = 100
patience = 10
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

print("Starting training...")
for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += y_batch.size(0)
        train_correct += (predicted == y_batch).sum().item()
    
    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += y_batch.size(0)
            val_correct += (predicted == y_batch).sum().item()
    
    avg_val_loss = val_loss / len(test_loader)
    val_accuracy = val_correct / val_total
    
    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(train_accuracy)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(val_accuracy)
    
    print(f'Epoch {epoch+1}/{epochs} - '
          f'accuracy: {train_accuracy:.4f} - loss: {avg_train_loss:.4f} - '
          f'val_accuracy: {val_accuracy:.4f} - val_loss: {avg_val_loss:.4f}')
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# ==================== Evaluation ====================
model.eval()
all_preds = []
all_probs = []

with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

y_pred = np.array(all_preds)
y_pred_probs = np.array(all_probs)

from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==================== Plot Training History ====================
training_acc = history['train_acc']
val_acc = history['val_acc']

EPOCHS = len(training_acc)
epoch_count = range(1, EPOCHS + 1)

plt.figure()
plt.plot(epoch_count, training_acc, "r--")
plt.plot(epoch_count, val_acc, "b-")
plt.legend(["LSTM Training Accuracy", "LSTM Val Accuracy"])
plt.xlabel("Epoch")
plt.ylabel("LSTM Accuracy")
plt.show()

# ==================== Save Model ====================
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_classes': num_classes,
        'dropout': dropout
    }
}, "lstm_model.pth")
print("Model saved to lstm_model.pth")
