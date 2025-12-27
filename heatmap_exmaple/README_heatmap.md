# Video Motion and Heatmap
*10/12/2025*

## Abtraction
Python based computer vision utility for motion density and activity over time, with the extraction of video frames, using cv2. Designed for the foundatoin of security style motion analysis.

## Methodology
- Extracts Frames at a targetted FPS with optional downscaling for maximum storage optimisation
- Coverts frames to greyscale to compute frame to frame difference, removing the shading differences and just focussing on pure movement.
- Creates a spatial heatmap of the overall image and then a temporal heatmap where movement happens

## Functions & Data
**Single video -> report of spatial motion heatmap & temporal motion activity**
**Functions: frame_extraction, heatmaps, load_frames**

## Use cases
- Basic security/surveillance
- Temporal motion hotspot identificiation for activity patterns
- Preprocessing for useful sections of videos, rapid exploration of long videos

## Benefits
- Lightweight, quick and interpreatable.
- No AI or training of models needed or additional data.

## Limitations
- Assumes static camera, and not for live video streams yet
- Sensitive to lighting changes
- No object/region classifciation (purpose of handstand and jump analysis)
