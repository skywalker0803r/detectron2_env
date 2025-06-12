# detect_landing.py
# 👉 從出手點 frame 向前推估落地幀，並輸出 JSON + 圖片

import os
import json
import cv2
from pose_utils_detectron2 import load_pose_sequence

OUTPUT_DIR = "output_landing"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "landing_frame.json")


def detect_landing(
    pose_sequence, release_frame, output_json=OUTPUT_JSON, back_offset=9
):
    """
    從出手點 frame 向前推 back_offset 幀作為落地幀，並輸出落地骨架與圖片
    - pose_sequence: 由 load_pose_sequence() 輸出的骨架序列
    - release_frame: 出手點的 frame 編號
    - output_json: 儲存 landing_frame.json 的路徑
    - back_offset: 向前推估的幀數，預設 9
    回傳：landing_frame（int）
    """
    release_index = next(
        (i for i, item in enumerate(pose_sequence) if item["frame"] == release_frame),
        None,
    )
    if release_index is None:
        raise ValueError(f"❌ 找不到 release_frame = {release_frame} 的對應資料")

    target_index = release_index - back_offset
    if target_index < 0:
        raise ValueError(f"❌ 推估 index = {target_index} 超出序列長度")

    landing_item = pose_sequence[target_index]
    landing_frame = landing_item["frame"]
    keypoints = landing_item["keypoints"]

    # 儲存 annotated 圖片
    image_dir = "output_detectron2_first_person_tracked"
    image_path = os.path.join(image_dir, f"frame_{landing_frame:05d}_annotated.jpg")
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        save_path = os.path.join(OUTPUT_DIR, f"{landing_frame}.jpg")
        cv2.imwrite(save_path, img)

    # 儲存 JSON
    result = {
        "landing_frame": landing_frame,
        "from_release_frame": release_frame,
        "keypoints": keypoints.tolist(),
    }
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"🦶 落地點 Frame: {landing_frame} → 儲存至 {output_json}")
    return landing_frame


# ✅ 可獨立執行（選配）
if __name__ == "__main__":
    from pose_utils_detectron2 import load_pose_sequence

    pose_sequence = load_pose_sequence("output_detectron2_first_person_tracked")
    release_json = "output_release/release_frame.json"
    with open(release_json, "r") as f:
        release_frame = json.load(f)["release_frame"]

    detect_landing(pose_sequence, release_frame)
