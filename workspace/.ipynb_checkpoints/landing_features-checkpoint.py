# landing_feat.py
# 👉 Detectron2 版：踏地點特徵分析

import numpy as np
from pose_utils_detectron2 import (
    calculate_pixel_angle_from_points,
    get_pose_window,
    visualize_pose_features,
)


def detect_landing_features(pose_sequence, landing_frame):
    """
    輸出 landing 點附近的特徵（跨步角、腳穩定度、展髖角）。
    - pose_sequence: list，每幀含 'frame' 與 'keypoints'
    - landing_frame: int，落地關鍵幀編號
    回傳 dict 特徵值
    """

    # COCO keypoints 順序
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6

    window = get_pose_window(pose_sequence, landing_frame, window_size=3)
    features = []

    for item in window:
        kpts = item["keypoints"]

        if (
            np.min(
                kpts[
                    [
                        LEFT_HIP,
                        RIGHT_HIP,
                        LEFT_ANKLE,
                        RIGHT_ANKLE,
                        LEFT_SHOULDER,
                        RIGHT_SHOULDER,
                    ],
                    2,
                ]
            )
            < 0.3
        ):
            continue

        # 特徵一：跨步角（左髖 - 左踝 - 右髖）
        hip_angle = calculate_pixel_angle_from_points(
            kpts[LEFT_HIP][:2], kpts[LEFT_ANKLE][:2], kpts[RIGHT_HIP][:2]
        )

        # 特徵二：腳穩定性（左右腳 heel Y 差）
        foot_y_diff = abs(kpts[LEFT_ANKLE][1] - kpts[RIGHT_ANKLE][1])

        # 特徵三：展髖角（左肩 - 左髖 - 右髖）
        hip_opening = calculate_pixel_angle_from_points(
            kpts[LEFT_SHOULDER][:2], kpts[LEFT_HIP][:2], kpts[RIGHT_HIP][:2]
        )

        features.append(
            {
                "frame": item["frame"],
                "hip_angle": hip_angle,
                "foot_y_diff": foot_y_diff,
                "hip_opening": hip_opening,
            }
        )

    if not features:
        print("⚠️ 無法計算踏地特徵，資料不足")
        return {}

    # 找最穩定的幀（腳 Y 差最小）
    best = min(features, key=lambda f: f["foot_y_diff"])
    print("🧾 踏地特徵：")
    for key, value in best.items():
        if key != "frame":
            print(f"  {key}: {value:.2f}")

    return best


if __name__ == "__main__":
    from pose_utils_detectron2 import load_pose_sequence

    pose_sequence = load_pose_sequence("output_detectron2_first_person_tracked")
    landing_frame = 123  # 測試用，請改成實際數值
    detect_landing_features(pose_sequence, landing_frame)
