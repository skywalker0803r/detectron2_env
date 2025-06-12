"""
shoulder_feat.py
👉 偵測肩膀開展特徵（Detectron2），分析出手前肩膀展開的程度與穩定性。
"""

import numpy as np
from pose_utils_detectron2 import get_pose_window, calculate_pixel_angle_from_points

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
LEFT_WRIST = 9
RIGHT_WRIST = 10


def extract_shoulder_features(pose_sequence, shoulder_frame, landing_frame):
    """
    提取肩膀展開幀附近的特徵：
    - shoulder_angle：肩膀開展角度（左肩-右肩-左髖）
    - hand_symmetry_y：雙手在 y 軸的對稱程度
    - shoulder_open_speed：肩膀開展角度變化速度
    """
    window = get_pose_window(pose_sequence, shoulder_frame, window_size=3)
    if not window:
        raise ValueError(f"❌ 找不到 shoulder_frame = {shoulder_frame} 的骨架資料")

    angles = []
    symmetries = []

    for item in window:
        keypoints = item["keypoints"]

        def valid(i):
            return keypoints[i, 2] > 0.3

        if not all(
            valid(i)
            for i in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, LEFT_WRIST, RIGHT_WRIST]
        ):
            continue

        angle = calculate_pixel_angle_from_points(
            keypoints[LEFT_SHOULDER, :2],
            keypoints[RIGHT_SHOULDER, :2],
            keypoints[LEFT_HIP, :2],
        )
        symmetry_y = abs(keypoints[LEFT_WRIST, 1] - keypoints[RIGHT_WRIST, 1])

        angles.append(angle)
        symmetries.append(symmetry_y)

    if not angles:
        raise ValueError("❌ 所有幀的肩膀關鍵點皆無效")

    shoulder_angle = np.mean(angles)
    hand_symmetry_y = np.mean(symmetries)
    shoulder_open_speed = np.std(angles)

    return {
        "shoulder_angle": round(shoulder_angle, 1),
        "hand_symmetry_y": round(hand_symmetry_y, 1),
        "shoulder_open_speed": round(shoulder_open_speed, 2),
    }


if __name__ == "__main__":
    import json
    from pose_utils_detectron2 import load_pose_sequence

    pose_sequence = load_pose_sequence("output_detectron2_first_person_tracked")

    with open("output_shoulder/shoulder_frame.json", "r") as f:
        shoulder_frame = json.load(f)["shoulder_frame"]
    with open("output_landing/landing_frame.json", "r") as f:
        landing_frame = json.load(f)["landing_frame"]

    feats = extract_shoulder_features(pose_sequence, shoulder_frame, landing_frame)
    print("🧍‍♂️ 肩膀特徵：")
    for k, v in feats.items():
        print(f"  {k}: {v}")
