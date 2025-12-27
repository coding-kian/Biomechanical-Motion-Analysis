import numpy as np, matplotlib.pyplot as plt
import os, cv2

def frame_extraction(frame_dir, video_path, downscale, quality):
    os.makedirs(frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path) # captured video
    fps, width, height = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*downscale), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*downscale)
    resize = lambda frame: cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    imwrite_params = [cv2.IMWRITE_JPEG_QUALITY, quality] 

    saved_paths = []    
    for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        flag, frame = cap.read() 
        if not flag: break
        if not i%round(fps/TARGET_FPS): continue
        path = os.path.join(frame_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(path, resize(frame), imwrite_params) # jpeg because png is lossless and this will be better for storage
        saved_paths.append(path)
    cap.release()

    print(f"Video FPS: {fps:.2f} @ {width/downscale}x{height/downscale}")
    print(f"Saved: {len(saved_paths)} {saved_paths[-1]}")


def heatmaps(frame_paths):
    prev_gray = None
    total_motion = 0.0
    vals = [] # motion value

    for path in frame_paths:
        gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            difference = cv2.absdiff(gray, prev_gray)
            total_motion += difference
            vals.append(difference.mean())
        prev_gray = gray

    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, gridspec_kw={"height_ratios": [10, 1]})

    ax1.imshow(total_motion/total_motion.max(), cmap="hot_r"); ax1.axis("off"); ax1.set_title("Movement density")
    ax2.imshow(np.asarray(vals)[None, :], aspect="auto", cmap="hot_r")
    ax2.set(title="Movement over time", xlabel="frame number", ylabel=[])


def load_frames(frame_dir) -> None:
    return sorted(os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.lower().endswith((".png",".jpg",".jpeg")))
    
if __name__ == "__main__":
    TARGET_FPS = 12
    downscale = 0.25 # 1.0 = original size, 0.5 = half resolution
    quality = 75 # quality: 0–100 (higher = better, bigger)
    FRAME_DIR = r"C:\Users\...\frames"
    frame_extraction(FRAME_DIR, video_path="videos_jump/manyjumps.mp4", downscale=downscale, quality=quality)
    frame_paths = load_frames(FRAME_DIR)
    print(f"Found {len(frame_paths)} frames.")
    heatmaps(frame_paths)
    plt.show()
