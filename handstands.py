from util_config import draw_label, initialising_video, labelling_video
import numpy as np, matplotlib.pyplot as plt, pandas as pd


def analyze_handstands(in_dir, out_dir, slowmo=1, downscale=1):
    frame_num = 0
    frames, all_poses = [], []
    cap, fps, out_size, out, pose = initialising_video(in_dir, out_dir, slowmo, downscale)
    while True:
        ok, frame = cap.read()
        if not ok: break
        joints, poses, current_landmarks, frame = labelling_video(out_size, downscale, pose, frame)
        if current_landmarks:
            frames.append(frame_num)
            all_poses.append(poses)            
            
        draw_label(frame, f"Frame {frame_num}", (10, 30), 0.9)
        out.write(frame)
        frame_num += 1

    cap.release() # closes the intial video
    out.release() # closes & saves the new video
    pose.close()
    print("Saved:", out_dir)

    frames = np.asarray(frames)
    shoulder = np.array([i.shoulder_deg for i in all_poses])
    elbow = np.array([i.elbow_deg for i in all_poses])
    knee = np.array([i.knee_deg for i in all_poses])
    hip = np.array([i.hip_deg for i in all_poses])

    if len(frames):
        fig, ax = plt.subplots(2, 2, figsize=(16, 9))
        plots = [(shoulder, "Shoulder angle", "deg"),
            (elbow, "Elbow angle", "deg"),
            (knee, "Knee angle", "deg"),
            (hip, "Hip angle", "deg")]

        for a, (y, title, ylab) in zip(ax.ravel(), plots):
            a.plot(frames, y)
            a.set(title=title, xlabel="Frame", ylabel=ylab); a.grid(True)

        

if __name__ == "__main__":
    dirr_ = "videos_handstand/"
    analyze_handstands(dirr_+"hspu.mp4", "_0hspu.mp4")
    plt.tight_layout()
    plt.show()


