# Biochechanical Motion Analysis
*10/02/2026*

## Introduction
Python based computer vision system for extracting **stable biomechanical signals** from sports videos. Converting noisy 2D videos into interpretable time-series suitable for optimisation and comparitive evaluation, with pose estimation from mediapie (trained on a large CNN of poses), looks at 33 points per frame.

Packages: cv2, mediapie, matplotlib, pandas & numpy.

## Objectives addressed
- Stablise noisy inputs for comparison and evaluation over time
- Provide geometric reasoning for relative position, magnitudes and angles between tracked points/poses/landmarks
- Deliver calibrated, comparable metrics for technique analysis in sports (dynamic videos)
- Preserve interpretability for debugging and optimisation
- The underlying motivation was application in real world infrastructure intelligence, for reliable conclusions from imperfect data


## Methodology.
- MediaPipe pose estimation (33 landmarks per frame) extracted from 2D landmark
- Geometric reconstruction of joint angles using vector geometry (cross and dot product)
- Mid-point to reduce asymetry and scale normalization for height of user via body scale.
- Temporal smoothing using EMAs to stablise signals before numerican differentiation for vertical velocity calculation

## Files & Data.
The system produces joint angles, annotated videos for validation from a single 2D video and a time series plot for comparison and optimsation of technique
**RGB Video Handstand video -> Annotated video & Report: Vertical velocity, knee angle, hip angle**
**RGB Video Jumping video -> Annotated video & Report: Maximum Hip Height, vertical velocity, knee angle, hip angle**

- `util_config.py` – Pose extraction, geometry, and video utilities  
- `jumping.py` – Joint stability/angles, jump height and vertical velocity analysis  
- `handstands.py` – Joint stability/angles and alignment during handstands


## Advantages
- Lightweight, fast and interpretable
- Gernalises across multiple movement patterns without additional training
- Annotated video and plots for clear technique and behaviour analysis

## Limitations & Improvements
- Depth inferred from 2D projection, leading to sensitive calibration of camera placement
- Assumes single subject/target, tracking may misalign


## Infrastructre Intelligence Relevence
- Extract stable signals from noisy telemetry
- Track system states and relationship over time
- Detect transitions and anomolies
- Evaluate performance probabllistically, through minimal dependencies.
- Interprate determinsitic metrics for optimisation

