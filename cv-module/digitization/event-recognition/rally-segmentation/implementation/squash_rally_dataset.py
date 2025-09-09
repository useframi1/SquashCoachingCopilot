import torch
from torch.utils.data import Dataset
import os
import cv2


class SquashRallyDataset(Dataset):
    def __init__(self, video_paths, annotations, sequence_length, transform=None):
        self.video_paths = video_paths
        self.annotations = annotations
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        """Prepare samples from annotations"""
        samples = []

        for video_id, anns in self.annotations.items():
            video_path = self._find_video_path(video_id)
            if not video_path:
                continue

            # Extract positive samples (rally starts)
            for ann in anns:
                if ann["label"] == "rally_start":
                    start_frame = ann["start_frame"]
                    end_frame = ann["end_frame"]

                    # Ensure we have enough frames
                    if end_frame - start_frame + 1 >= self.sequence_length:
                        samples.append(
                            {
                                "video_path": video_path,
                                "start_frame": start_frame,
                                "label": 1,
                            }
                        )

                if ann["label"] == "rally_end":
                    start_frame = ann["start_frame"]
                    end_frame = ann["end_frame"]

                    # Ensure we have enough frames
                    if end_frame - start_frame + 1 >= self.sequence_length:
                        samples.append(
                            {
                                "video_path": video_path,
                                "start_frame": start_frame,
                                "label": 2,
                            }
                        )

                # Extract negative samples (not rally starts)
                elif ann["label"] == "not_rally_start":
                    start_frame = ann["start_frame"]
                    end_frame = ann["end_frame"]

                    # Ensure we have enough frames
                    if end_frame - start_frame + 1 >= self.sequence_length:
                        samples.append(
                            {
                                "video_path": video_path,
                                "start_frame": start_frame,
                                "label": 0,
                            }
                        )

        return samples

    def _find_video_path(self, video_id):
        """Find the full path to a video given its ID"""
        for path in self.video_paths:
            if os.path.basename(path) == video_id:
                return path
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Load video sequence and return tensor"""
        sample = self.samples[idx]
        video_path = sample["video_path"]
        start_frame = sample["start_frame"]
        label = sample["label"]

        # Open video file
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Read sequence_length frames
        frames = []
        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                # If we run out of frames, duplicate the last frame
                if frames:
                    frames.append(frames[-1])
                else:
                    # This should not happen with proper annotations
                    raise ValueError(
                        f"Could not read first frame from {video_path} at position {start_frame}"
                    )
            else:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)

        cap.release()

        # Stack frames into a sequence tensor [sequence_length, channels, height, width]
        sequence = torch.stack(frames)

        return sequence, torch.tensor(label, dtype=torch.float32)
