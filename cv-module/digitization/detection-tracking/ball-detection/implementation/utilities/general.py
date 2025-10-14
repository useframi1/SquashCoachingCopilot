import torch
import time
import numpy as np
import torch.nn as nn
import cv2
from scipy.spatial import distance
from scipy.signal import medfilt, savgol_filter


def load_config(config_path="config.json"):
    """Load configuration from a JSON file."""
    import json

    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def train(model, train_loader, optimizer, device, epoch, max_iters=200):
    start_time = time.time()
    losses = []
    criterion = nn.CrossEntropyLoss()
    for iter_id, batch in enumerate(train_loader):
        optimizer.zero_grad()
        model.train()
        out = model(batch[0].float().to(device))
        gt = torch.tensor(batch[1], dtype=torch.long, device=device)
        loss = criterion(out, gt)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        end_time = time.time()
        duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
        print(
            "train | epoch = {}, iter = [{}|{}], loss = {}, time = {}".format(
                epoch, iter_id, max_iters, round(loss.item(), 6), duration
            )
        )
        losses.append(loss.item())

        if iter_id > max_iters - 1:
            break

    return np.mean(losses)


def validate(model, val_loader, device, epoch, min_dist=5):
    losses = []
    tp = [0, 0, 0, 0]
    fp = [0, 0, 0, 0]
    tn = [0, 0, 0, 0]
    fn = [0, 0, 0, 0]
    criterion = nn.CrossEntropyLoss()
    model.eval()
    for iter_id, batch in enumerate(val_loader):
        with torch.no_grad():
            out = model(batch[0].float().to(device))
            gt = torch.tensor(batch[1], dtype=torch.long, device=device)
            loss = criterion(out, gt)
            losses.append(loss.item())
            # metrics
            output = out.argmax(dim=1).detach().cpu().numpy()
            for i in range(len(output)):
                x_pred, y_pred = postprocess(output[i])
                x_gt = batch[2][i]
                y_gt = batch[3][i]
                vis = batch[4][i]
                if x_pred:
                    if vis != 0:
                        dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        if dst < min_dist:
                            tp[vis] += 1
                        else:
                            fp[vis] += 1
                    else:
                        fp[vis] += 1
                if not x_pred:
                    if vis != 0:
                        fn[vis] += 1
                    else:
                        tn[vis] += 1
            print(
                "val | epoch = {}, iter = [{}|{}], loss = {}, tp = {}, tn = {}, fp = {}, fn = {} ".format(
                    epoch,
                    iter_id,
                    len(val_loader),
                    round(np.mean(losses), 6),
                    sum(tp),
                    sum(tn),
                    sum(fp),
                    sum(fn),
                )
            )
    eps = 1e-15
    precision = sum(tp) / (sum(tp) + sum(fp) + eps)
    vc1 = tp[1] + fp[1] + tn[1] + fn[1]
    vc2 = tp[2] + fp[2] + tn[2] + fn[2]
    vc3 = tp[3] + fp[3] + tn[3] + fn[3]
    recall = sum(tp) / (vc1 + vc2 + vc3 + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    print("precision = {}".format(precision))
    print("recall = {}".format(recall))
    print("f1 = {}".format(f1))

    return np.mean(losses), precision, recall, f1


def postprocess(feature_map, scale=2):
    feature_map *= 255
    feature_map = feature_map.reshape((360, 640))
    feature_map = feature_map.astype(np.uint8)
    ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        heatmap,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1,
        param1=50,
        param2=2,
        minRadius=2,
        maxRadius=7,
    )
    x, y = None, None
    if circles is not None:
        if len(circles) == 1:
            x = circles[0][0][0] * scale
            y = circles[0][0][1] * scale
    return x, y


def smooth_positions(ball_positions, median_window=5, savgol_window=9, savgol_poly=3):
    """Apply two-stage smoothing to ball positions.

    Stage 1: Median filter to remove outlier spikes
    Stage 2: Savitzky-Golay filter for smooth curves

    Args:
        ball_positions: List of (x, y) tuples representing ball positions
        median_window: Window size for median filter (must be odd)
        savgol_window: Window size for Savitzky-Golay filter (must be odd)
        savgol_poly: Polynomial order for Savitzky-Golay filter

    Returns:
        List of smoothed (x, y) tuples
    """
    if len(ball_positions) < savgol_window:
        return ball_positions

    # Separate x and y coordinates
    x_coords = np.array(
        [pos[0] if pos[0] is not None else np.nan for pos in ball_positions]
    )
    y_coords = np.array(
        [pos[1] if pos[1] is not None else np.nan for pos in ball_positions]
    )

    # Interpolate missing values (NaNs)
    valid_x_indices = ~np.isnan(x_coords)
    valid_y_indices = ~np.isnan(y_coords)

    if np.sum(valid_x_indices) > 1:
        x_coords_interp = np.interp(
            np.arange(len(x_coords)),
            np.where(valid_x_indices)[0],
            x_coords[valid_x_indices],
        )
    else:
        x_coords_interp = x_coords

    if np.sum(valid_y_indices) > 1:
        y_coords_interp = np.interp(
            np.arange(len(y_coords)),
            np.where(valid_y_indices)[0],
            y_coords[valid_y_indices],
        )
    else:
        y_coords_interp = y_coords

    # Stage 1: Apply median filter to remove outlier spikes
    x_median = medfilt(x_coords_interp, kernel_size=median_window)
    y_median = medfilt(y_coords_interp, kernel_size=median_window)

    # Stage 2: Apply Savitzky-Golay filter for smooth curves
    x_smoothed = savgol_filter(
        x_median, window_length=savgol_window, polyorder=savgol_poly
    )
    y_smoothed = savgol_filter(
        y_median, window_length=savgol_window, polyorder=savgol_poly
    )

    # Convert back to list of tuples
    smoothed_positions = [(int(x), int(y)) for x, y in zip(x_smoothed, y_smoothed)]

    return smoothed_positions
