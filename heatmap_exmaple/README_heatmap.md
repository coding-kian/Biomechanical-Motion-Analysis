title: "Lightweight Video Motion Heatmap Analyzer"

summary: >
  A minimal computer vision utility for visualising motion density and activity
  over time in video streams. Designed as a lightweight, interpretable foundation
  for security-style motion analysis.

methodology:
  - Extract frames at a fixed target FPS with optional downscaling
  - Convert frames to grayscale and compute frame-to-frame differences
  - Accumulate motion into a spatial heatmap
  - Track per-frame motion intensity as a temporal signal

data_flow:
  input:
    - Single static-camera video file
  output:
    - Spatial motion density heatmap
    - Temporal motion activity plot

key_functions:
  - frame_extraction: Sample and store video frames efficiently
  - heatmaps: Compute spatial and temporal motion representations
  - load_frames: Load extracted frames in sorted order

advantages:
  - Extremely lightweight and fast
  - No training, models, or external datasets required
  - Interpretable pixel-level motion representation
  - Suitable for exploratory analysis and low-resource environments

limitations:
  - Assumes a static camera
  - Sensitive to lighting changes and camera noise
  - No object classification or tracking
  - Not suitable for complex multi-camera scenarios without extension

use_cases:
  - Basic security and surveillance analysis
  - Motion hotspot identification
  - Activity pattern visualisation
  - Preprocessing for more advanced video analytics
  - Rapid exploratory analysis of video data

future_improvements:
  - Background subtraction or adaptive thresholding
  - Region-of-interest masking
  - Event detection and alerting
  - Integration with real-time video streams
  - Export of motion metrics for downstream analytics
