"""
detect_release.py
👉 偵測出手幀（release frame），使用 Detectron2 輸出的 pixel keypoints。
"""

import os
import cv2
import numpy as np
import json
from pose_utils_detectron2 import calculate_pixel_angle_from_points

OUTPUT_DIR = "output_release"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "release_frame.json")


def detect_release(pose_sequence, output_json=OUTPUT_JSON):
    """
    根據關鍵點條件與角度計算出手幀。
    - pose_sequence: list，每幀包含 'frame' 和 'keypoints'
    回傳最佳出手 frame 編號
    """

    candidate_frames = []

    for item in pose_sequence:
        frame_idx = item["frame"]
        keypoints = item["keypoints"]  # (17, 3)
        if keypoints.shape[0] < 17:
            continue

        # 關鍵點信心值過濾
        if np.min(keypoints[[11, 12, 14, 16], 2]) < 0.3:
            continue

        right_shoulder = keypoints[12]
        left_shoulder = keypoints[11]
        right_elbow = keypoints[14]
        right_wrist = keypoints[16]

        # 基本量
        shoulder_dist = abs(right_shoulder[0] - left_shoulder[0])
        print(frame_idx, ":", shoulder_dist, right_wrist[1], right_shoulder[1])

        elbow_angle = calculate_pixel_angle_from_points(
            right_wrist[:2], right_elbow[:2], right_shoulder[:2]
        )

        shoulder_vec = np.array(
            [right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1]]
        )
        horizontal_vec = np.array([1, 0])
        cos_theta = np.dot(shoulder_vec, horizontal_vec) / np.linalg.norm(shoulder_vec)
        shoulder_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        # 出手條件
        if shoulder_dist > 25 and right_wrist[1] > right_shoulder[1]:
            candidate_frames.append(
                {
                    "frame": frame_idx,
                    "elbow_angle": elbow_angle,
                    "shoulder_angle": shoulder_angle,
                    "shoulder_dist": shoulder_dist,
                }
            )
        # 出手條件不滿足
        else:
            print('出手條件不滿足')

    if not candidate_frames:
        print("⚠️ 沒有符合條件的畫面")
        return None

    # 選出 shoulder_angle 最小的前三幀
    top3 = sorted(candidate_frames, key=lambda f: f["shoulder_angle"])[:3]
    print(top3)

    def compute_score(f):
        return -1.5 * f["shoulder_angle"] + 0.8 * f["elbow_angle"]

    best_frame = max(top3, key=compute_score)
    frame_id = best_frame["frame"]

    # 儲存 json
    with open(output_json, "w") as f:
        json.dump({"release_frame": frame_id}, f, indent=2)
    print(f"📸 已儲存出手資訊：{output_json}")

    return frame_id


if __name__ == "__main__":
    from pose_utils_detectron2 import load_pose_sequence

    pose_sequence = load_pose_sequence("output_detectron2_first_person_tracked")
    result = detect_release(pose_sequence)
    print(f"✅ 出手偵測結果：{result}")
