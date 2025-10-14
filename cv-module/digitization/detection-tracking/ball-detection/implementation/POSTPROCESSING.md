# Simple Postprocessing for Wall Hit Detection

## Goal
Clean the ball position curve (especially the **Y coordinate**) so you can clearly see when the ball changes direction (hits the wall).

## How It Works

### 3 Simple Steps:

1. **Remove Outliers** → Gets rid of bad detections that create jumps
2. **Fill Missing Values** → Fills any gaps with interpolation
3. **Smooth the Curve** → Makes the position trajectory clean and clear

That's it!

## Configuration

Edit `config.json`:

```json
{
  "postprocessing": {
    "enabled": true,

    "outlier_detection": {
      "window": 7,        // How many neighbors to compare against
      "threshold": 2.5    // How far from neighbors = outlier (in std devs)
    },

    "smoothing": {
      "enabled": true,
      "method": "savgol",  // or "median"
      "window": 7,         // Smoothing window size (smaller = less smooth)
      "poly": 2            // Polynomial order (2 or 3)
    }
  }
}
```

## What Each Parameter Does

### Outlier Detection
- **window**: How many nearby frames to compare each position against (default: 7)
- **threshold**: How many standard deviations away = outlier (default: 2.5)
  - Lower = more aggressive outlier removal
  - Higher = only removes obvious bad detections

### Smoothing
- **method**:
  - `"savgol"` (default) - Savitzky-Golay filter, preserves peaks/valleys well
  - `"median"` - Median filter, removes spikes aggressively

- **window**: Size of smoothing window (default: 7)
  - Smaller (5) = less smooth, more detail
  - Larger (9-11) = more smooth, cleaner

- **poly**: Polynomial order for savgol (default: 2)
  - Use 2 for simpler curves
  - Use 3 for more complex trajectories

## Tuning for Wall Hit Detection

### For Clear Y-Direction Changes

When the ball hits the front wall:
- Y coordinate **decreases** (ball moving toward wall)
- Then **increases** (ball bouncing back)

You want smoothing that **preserves this direction change** but removes noise.

**Recommended settings:**
```json
{
  "smoothing": {
    "method": "savgol",
    "window": 7,     // Good balance
    "poly": 2        // Preserves direction changes
  }
}
```

### If the curve is too noisy:
- Increase `window` to 9 or 11
- Or try `method: "median"` for aggressive noise removal

### If wall hits are getting smoothed out:
- Decrease `window` to 5
- Use `poly: 3` for more responsive savgol

### If you have bad detections creating jumps:
- Decrease outlier `threshold` to 2.0 (more aggressive)
- Increase outlier `window` to 9

## Example: Wall Hit Detection

After postprocessing, the Y coordinate should look like a clean curve with clear peaks and valleys.

To detect wall hits, you can then:
1. Calculate derivative of Y: `dy = y[i] - y[i-1]`
2. Find where dy changes sign (from negative to positive)
3. That's a wall hit!

```python
# Simple wall hit detection
wall_hits = []
for i in range(1, len(positions)-1):
    dy_before = positions[i][1] - positions[i-1][1]
    dy_after = positions[i+1][1] - positions[i][1]

    # Y was decreasing, now increasing = wall hit
    if dy_before < 0 and dy_after > 0:
        wall_hits.append(i)
```

## Testing Your Settings

1. Run: `python3 evaluator.py`
2. Look at `tracking_results/video_name/video_name_positions.png`
3. Check if Y coordinate curve is:
   - Smooth enough (no jitter)
   - But still shows clear peaks and valleys

If yes, you're ready for wall hit detection!
