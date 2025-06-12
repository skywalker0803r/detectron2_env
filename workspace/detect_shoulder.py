# detect_shoulder.py
# 👉 偵測投手肩膀最打開的幀，依據手腕高度啟動、X 軸肩寬、像素肩角進行評估

import os
import cv2
import json
import numpy as np
from pose_utils_detectron2 import load_pose_sequence, calculate_pixel_angle_from_points

# --- 輸出路徑設定 ---
OUTPUT_DIR = "output_shoulder"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "shoulder_frame.json")

# --- COCO 關鍵點索引 ---
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_WRIST = 10


def detect_shoulder(
    pose_sequence,
    release_frame,
    output_json=OUTPUT_JSON,
    image_dir="output_detectron2_first_person_tracked",
):
    """
    偵測肩膀最打開的幀：
    - 起始：右手腕高於右肩（代表投球已啟動）
    - 條件：右手腕仍在右肩左側且未落下
    - 評分：先取肩膀水平距離最大的前3幀，再從中挑肩角最大的幀

    回傳：
    - shoulder_frame（int）
    """
    candidate_list = []
    start_found = False

    for item in pose_sequence:
        frame_idx = item["frame"]
        if frame_idx > release_frame:
            break

        keypoints = item["keypoints"]
        if (
            np.min(keypoints[[LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_WRIST], 2])
            < 0.3
        ):
            continue

        l_sh = keypoints[LEFT_SHOULDER][:2]
        r_sh = keypoints[RIGHT_SHOULDER][:2]
        l_hip = keypoints[LEFT_HIP][:2]
        r_wr = keypoints[RIGHT_WRIST][:2]

        # 起始條件：右手腕高於右肩
        if not start_found:
            if r_wr[1] < r_sh[1]:
                start_found = True
            else:
                continue

        # 排除：右手腕落下 或 手腕已超過肩膀
        if r_wr[0] > r_sh[0] or r_wr[1] >= r_sh[1]:
            continue

        # 肩膀開啟角度（l_sh - r_sh - l_hip）
        angle = calculate_pixel_angle_from_points(l_sh, r_sh, l_hip)
        shoulder_distance = abs(r_sh[0] - l_sh[0])
        candidate_list.append((angle, shoulder_distance, frame_idx))

    if not candidate_list:
        raise ValueError("❌ 無法找到符合條件的肩膀開啟幀")

    # 取肩膀 X 軸距離最大的前三名 → 選角度最大者
    top3 = sorted(candidate_list, key=lambda x: -x[1])[:3]
    best = max(top3, key=lambda x: x[0])
    shoulder_frame = best[2]

    # 儲存圖像
    img_path = os.path.join(image_dir, f"frame_{shoulder_frame:05d}_annotated.jpg")
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        out_path = os.path.join(OUTPUT_DIR, f"shoulder_best_frame_{shoulder_frame}.jpg")
        cv2.imwrite(out_path, img)
        print(f"📸 肩膀最開圖片已儲存：{out_path}")

    # 儲存 JSON
    result = {"shoulder_frame": shoulder_frame}
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"🧍‍♂️ 肩膀最開 Frame: {shoulder_frame} → 已儲存至 {output_json}")
    return shoulder_frame


# ✅ 可單獨執行測試
if __name__ == "__main__":
    pose_sequence = load_pose_sequence("output_detectron2_first_person_tracked")
    with open("output_release/release_frame.json", "r") as f:
        release_frame = json.load(f)["release_frame"]
    detect_shoulder(pose_sequence, release_frame)
