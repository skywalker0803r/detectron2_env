"""
release_features.py
👉 擷取出手幀的特徵（Detectron2 keypoints），分析出手肘角度、肩膀展開程度與穩定性。
"""

import numpy as np
from pose_utils_detectron2 import get_pose_window, calculate_pixel_angle_from_points

RIGHT_SHOULDER = 6
RIGHT_ELBOW = 8
RIGHT_WRIST = 10
LEFT_SHOULDER = 5


def extract_release_features(pose_sequence, release_frame):
    """
    擷取出手幀 ±3 幀的特徵：
    - elbow_angle：右手肘角度（腕-肘-肩）
    - shoulder_angle：肩膀水平角度（左肩-右肩與 x 軸夾角）
    - elbow_stability：肘角度標準差
    - shoulder_stability：肩膀角度標準差
    """
    window = get_pose_window(pose_sequence, release_frame, window_size=3)
    if not window:
        raise ValueError(f"❌ 找不到 release_frame = {release_frame} 的骨架資料")

    elbow_angles = []
    shoulder_angles = []

    for item in window:
        kpts = item["keypoints"]

        def valid(i):
            return kpts[i, 2] > 0.3

        if not all(
            valid(i) for i in [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, LEFT_SHOULDER]
        ):
            continue

        # 肘角度：右腕 - 右肘 - 右肩
        elbow = calculate_pixel_angle_from_points(
            kpts[RIGHT_WRIST, :2], kpts[RIGHT_ELBOW, :2], kpts[RIGHT_SHOULDER, :2]
        )
        elbow_angles.append(elbow)

        # 肩膀角度：右肩 - 左肩 與水平軸夾角
        shoulder_vec = kpts[RIGHT_SHOULDER, :2] - kpts[LEFT_SHOULDER, :2]
        horizontal = np.array([1, 0])
        cos_theta = np.dot(shoulder_vec, horizontal) / np.linalg.norm(shoulder_vec)
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        shoulder_angles.append(angle)

    if not elbow_angles:
        raise ValueError("❌ 無法取得有效的肘角與肩角資料")

    return {
        "elbow_angle": round(np.mean(elbow_angles), 1),
        "shoulder_angle": round(np.mean(shoulder_angles), 1),
        "elbow_stability": round(np.std(elbow_angles), 2),
        "shoulder_stability": round(np.std(shoulder_angles), 2),
    }


if __name__ == "__main__":
    import json
    from pose_utils_detectron2 import load_pose_sequence

    pose_sequence = load_pose_sequence("output_detectron2_first_person_tracked")

    with open("output_release/release_frame.json", "r") as f:
        release_frame = json.load(f)["release_frame"]

    feats = extract_release_features(pose_sequence, release_frame)
    print("🏌️‍♂️ 出手特徵：")
    for k, v in feats.items():
        print(f"  {k}: {v}")
