import numpy as np
import cv2
from ultralytics import YOLO
import torch
from lstm_model import LSTMModel
import pickle  # For loading the scaler

RELEVANT_IDX = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
RELEVANT_NAMES = [
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

# Load models
keypoint_model = YOLO("yolov8n-pose.pt")  # YOLOv8 nano pose model
model = LSTMModel(24, 4, 3)
model.load_state_dict(torch.load("lstm_model.pt"))
model.eval()

# Load the fitted MinMaxScaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Parameters
sequence_length = 16  # Your LSTM input sequence length
num_features = 24  # Number of keypoints features
stroke_classes = ["forehand", "backhand", "neither"]


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Initialize buffers for both players
    player1_buffer = []
    player2_buffer = []

    # Results storage
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = keypoint_model.track(
            source=frame, persist=True, conf=0.6, verbose=False
        )
        result = results[0]
        frame_people = []

        keypoints_data = result.keypoints

        if keypoints_data is not None:
            for i, person_keypoints in enumerate(keypoints_data.data):
                if len(person_keypoints) <= max(RELEVANT_IDX):
                    continue  # Skip if not enough keypoints

                person_data = []

                for j, idx in enumerate(RELEVANT_IDX):
                    kp = person_keypoints[idx].cpu().numpy().tolist()
                    person_data.append(kp[0])
                    person_data.append(kp[1])

                frame_people.append(person_data)

        # Add to buffers
        if len(frame_people) < 2:
            continue
        player1_buffer.append(frame_people[0])
        player2_buffer.append(frame_people[1])

        # Keep buffers at sequence_length
        if len(player1_buffer) > sequence_length:
            player1_buffer.pop(0)
        if len(player2_buffer) > sequence_length:
            player2_buffer.pop(0)

        # Only make predictions when we have enough frames
        if len(player1_buffer) == sequence_length:
            # Prepare input for LSTM model for player 1
            player1_seq = np.array(player1_buffer)

            # Apply the scaler to each frame in the sequence
            # Reshape to 2D for scaling (samples*sequence_length, features)
            player1_seq_2d = player1_seq.reshape(-1, num_features)
            player1_seq_scaled = scaler.transform(player1_seq_2d)

            # Reshape back to 3D for LSTM (1, sequence_length, features)
            player1_input = player1_seq_scaled.reshape(1, sequence_length, num_features)
            player1_input = torch.tensor(player1_input, dtype=torch.float32)

            # Get prediction for player 1
            with torch.no_grad():
                player1_output = model(player1_input)
                player1_pred = torch.argmax(player1_output, dim=1).item()

            # Prepare input for LSTM model for player 2
            player2_seq = np.array(player2_buffer)

            # Apply the scaler to each frame in the sequence
            player2_seq_2d = player2_seq.reshape(-1, num_features)
            player2_seq_scaled = scaler.transform(player2_seq_2d)

            # Reshape back to 3D for LSTM
            player2_input = player2_seq_scaled.reshape(1, sequence_length, num_features)
            player2_input = torch.tensor(player2_input, dtype=torch.float32)

            # Get prediction for player 2
            with torch.no_grad():
                player2_output = model(player2_input)
                player2_pred = torch.argmax(player2_output, dim=1).item()

            # Store results
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
            results.append(
                {
                    "frame": int(frame_number),
                    "player1": stroke_classes[player1_pred],
                    "player2": stroke_classes[player2_pred],
                }
            )

            # Visualize results on frame (optional)
            cv2.putText(
                frame,
                f"Player 1: {stroke_classes[player1_pred]}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Player 2: {stroke_classes[player2_pred]}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Display frame
            cv2.imshow("Stroke Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    return results


# Call the function with your video
stroke_detections = process_video("test.mp4")
