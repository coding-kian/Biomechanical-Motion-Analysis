from util_config import draw_label, initialising_video, labelling_video
import numpy as np, matplotlib.pyplot as plt, pandas as pd
INCH_TO_M = 0.0254

def analyze_jumps(in_dir, out_dir, slowmo=0.5, downscale=1, height_inches=69, smoothing_range=12):
    hip_poi, onground_poi = [], [] # these are the reference points
    scale_samples, all_hip_heights = [], []

    frame_num = 0
    frames, all_poses = [], []
    cap, fps, out_size, out, pose = initialising_video(in_dir, out_dir, slowmo, downscale)
    while True:
        ok, frame = cap.read()
        if not ok: break
        joints, poses, current_landmarks, frame = labelling_video(out_size, downscale, pose, frame)

        if current_landmarks: # for the height from the ground and velocity
            frames.append(frame_num)
            all_poses.append(poses)
            max_foot_location = np.max([joints["l_heel"][1], joints["r_heel"][1]]) # Only need y axis (vertical velocity) just take index 1
            pixel_to_inch_ratio = height_inches/max(1e-6, max_foot_location-joints["nose"][1])
             

            if len(onground_poi) < smoothing_range:
                scale_samples.append(pixel_to_inch_ratio)
                hip_poi.append(joints["hip_mid"][1])
                onground_poi.append(max_foot_location)

            pixel_to_inch = float(np.mean(scale_samples)) if scale_samples else float(pixel_to_inch_ratio)
            hip_location = float(np.mean(hip_poi)) if hip_poi else float(joints["hip_mid"][1])
            floor_location = float(np.mean(onground_poi)) if onground_poi else float(max_foot_location)
            hip_height = (hip_location - joints["hip_mid"][1]) * pixel_to_inch # hip height to inches 
            left_off_in  = (floor_location - joints["l_heel"][1])  * pixel_to_inch
            right_off_in = (floor_location - joints["r_heel"][1]) * pixel_to_inch # just use ankle not heel

            draw_label(frame, f"Height {hip_height:+.1f} Inch", (10, 70))
            draw_label(frame, f"Left Foot {left_off_in:+.1f} Inch.  Right Foot:{right_off_in:+.1f} Inch", (10, 100))
            all_hip_heights.append(hip_height)

        draw_label(frame, f"Frame {frame_num}", (10, 30), 0.9)
        out.write(frame)
        frame_num += 1

    cap.release() # closes the intial video
    out.release() # closes & saves the new video
    pose.close()
    print("Saved:", out_dir)

    frames = np.asarray(frames)
    shoulder = np.array([i.shoulder_deg for i in all_poses])
    knee = np.array([i.knee_deg for i in all_poses])
    hip = np.array([i.hip_deg for i in all_poses])

    hip_height_average = pd.Series(all_hip_heights).ewm(span=smoothing_range).mean().to_numpy() # numpy dosnt have emas so pandas
    vertical_velocity = np.gradient(hip_height_average*INCH_TO_M, 1/fps)

    if len(frames):
        fig, ax = plt.subplots(2, 2, figsize=(16, 9))
        plots = [(hip_height_average, "Hip Height EMA", "Inches"),
                (vertical_velocity, "Vertical Velocity", "m/s"),
                (knee, "Knee angle", "deg"),
                (hip, "Hip&Shoulder angle", "deg")]

        for a, (y, title, ylab) in zip(ax.ravel(), plots):
            if title == "Hip&Shoulder angle":
                a.plot(frames, shoulder, label="Arms"); a.plot(frames, hip, label="Hip"); a.legend() # just so it is a double plot
            else: a.plot(frames, y)
            a.set(title=title, xlabel="Frame", ylabel=ylab); a.grid(True)
            

if __name__ == "__main__":
    dirr_ = "videos_jump/"
    analyze_jumps(dirr_+"flip.mp4", "_0jumps.mp4", slowmo=0.5)
    plt.tight_layout()
    plt.show()

    
