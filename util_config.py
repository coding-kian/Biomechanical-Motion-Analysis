import cv2
import numpy as np, matplotlib.pyplot as plt, mediapipe as mp
from dataclasses import dataclass

def plotting(x_, y_, x_label, y_label, title) -> None:
    plt.figure(); plt.plot(x_, y_)
    plt.xlabel(x_label); plt.ylabel(y_label); plt.title(title); plt.grid(True)


def draw_label(img, text, xy, scale=0.6, color=(0, 255, 255)) -> None:
    x, y = map(int, xy)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def angle_deg(p1, pivot, p2) -> float:
    v1 = (p1 - pivot)
    v2 = (p2 - pivot)
    return abs(np.degrees(np.arctan2(np.cross(v1, v2), np.dot(v1, v2)))) # cross then dot


def landmark_magnitude(landmarks, lm_enum, w: int, h: int) -> np.ndarray:
    p = landmarks[lm_enum.value]
    return np.array([p.x * w, p.y * h])


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5


@dataclass
class FrameMetrics:
    shoulder_deg: float
    elbow_deg: float
    knee_deg: float
    hip_deg: float


def landmark_centre(joints, left_triplet, right_triplet) -> float:
    return 0.5*(angle_deg(joints[left_triplet[0]], joints[left_triplet[1]], joints[left_triplet[2]]) + 
            angle_deg(joints[right_triplet[0]], joints[right_triplet[1]], joints[right_triplet[2]]))


def calculate_poses(joints: dict) -> FrameMetrics:
    return FrameMetrics(
        landmark_centre(joints, ("l_hip", "l_shoulder", "l_wrist"), ("r_hip", "r_shoulder", "r_wrist")),
        landmark_centre(joints, ("l_shoulder", "l_elbow", "l_wrist"), ("r_shoulder", "r_elbow", "r_wrist")),
        landmark_centre(joints, ("l_hip", "l_knee", "l_ankle"), ("r_hip", "r_knee", "r_ankle")),
        landmark_centre(joints, ("l_shoulder", "l_hip", "l_knee"), ("r_shoulder", "r_hip", "r_knee")))


def all_landmarks(landmarks, lm, w: int, h: int) -> dict: # lm=landmarks, which are the body poi (joints)
    joint_map = {"nose": lm.NOSE,
        "l_shoulder": lm.LEFT_SHOULDER, "r_shoulder": lm.RIGHT_SHOULDER,
        "l_hip": lm.LEFT_HIP, "r_hip": lm.RIGHT_HIP,
        "l_elbow": lm.LEFT_ELBOW, "r_elbow": lm.RIGHT_ELBOW,
        "l_wrist": lm.LEFT_WRIST, "r_wrist": lm.RIGHT_WRIST,
        "l_knee": lm.LEFT_KNEE, "r_knee": lm.RIGHT_KNEE,
        "l_heel": lm.LEFT_HEEL, "r_heel": lm.RIGHT_HEEL,
        "l_ankle": lm.LEFT_ANKLE, "r_ankle": lm.RIGHT_ANKLE}
    joints = {name: landmark_magnitude(landmarks, enum, w, h) for name, enum in joint_map.items()}

    joints["wrist_mid"] = midpoint(joints["l_wrist"], joints["r_wrist"])
    joints["shoulder_mid"] = midpoint(joints["l_shoulder"], joints["r_shoulder"])
    joints["hip_mid"] = midpoint(joints["l_hip"], joints["r_hip"])
    joints["knee_mid"]     = midpoint(joints["l_knee"], joints["r_knee"])
    joints["elbow_mid"] = midpoint(joints["l_elbow"], joints["r_elbow"])
    
    return joints


mp_pose, mp_draw = mp.solutions.pose, mp.solutions.drawing_utils

def initialising_video(in_dir, out_dir, slowmo=0.5, downscale=1):
    cap = cv2.VideoCapture(in_dir) # cature
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*downscale),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*downscale)) # swaps height and width since rotated
    out = cv2.VideoWriter(out_dir, cv2.VideoWriter_fourcc(*"mp4v"), fps*slowmo, out_size)
    pose = mp_pose.Pose() # higher detection: probability to track START tracking. Higher tracking: probability to KEEP tracking
    return cap, fps, out_size, out, pose


def labelling_video(out_size, downscale, pose, frame):
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) 
    if downscale != 1.0: frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2] # height and width
    
    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    current_landmarks = res.pose_landmarks
    if current_landmarks:
        mp_draw.draw_landmarks(frame, current_landmarks, mp_pose.POSE_CONNECTIONS)
        joints = all_landmarks(current_landmarks.landmark, mp_pose.PoseLandmark, w, h)

        poses = calculate_poses(joints)        
        draw_label(frame, f"Shoulder {poses.shoulder_deg:.0f}*", joints["shoulder_mid"], 0.75)
        draw_label(frame, f"Elbow {poses.elbow_deg:.0f}*", joints["elbow_mid"], 0.75)
        draw_label(frame, f"Knee {poses.knee_deg:.0f}*", joints["knee_mid"], 0.75)
        draw_label(frame, f"Hip {poses.hip_deg:.0f}*", joints["hip_mid"], 0.75)

        return joints, poses, current_landmarks, frame
    return None, None, None, frame

