# analyze_pitch_from_detectron2.py
# 👉 全自動流程：影片 → Detectron2 → 三動作偵測 → 特徵分析

import json
from pose_utils_detectron2 import load_pose_sequence
from pose_detectron2 import run_detectron2

from detect_release import detect_release
from detect_landing import detect_landing
from detect_shoulder import detect_shoulder

from landing_features import detect_landing_features
from shoulder_features import extract_shoulder_features
from release_features import extract_release_features

# --- 設定輸入與輸出資料夾 ---
VIDEO_PATH = "data/Yu_Darvish_FF_videos_4S/pitch_0001.mp4"
KEYPOINT_DIR = "output_detectron2_first_person_tracked"


def main():
    print(f"📹 處理影片：{VIDEO_PATH}")

    # ✅ 步驟 1：執行 Detectron2 偵測並輸出骨架資料
    run_detectron2(video_path=VIDEO_PATH, output_dir=KEYPOINT_DIR)

    # ✅ 步驟 2：載入骨架序列
    pose_sequence = load_pose_sequence(KEYPOINT_DIR)

    # ✅ 步驟 3：偵測三個關鍵幀
    release_frame = detect_release(pose_sequence)
    landing_frame = detect_landing(pose_sequence, release_frame)
    shoulder_frame = detect_shoulder(pose_sequence, release_frame)

    # ✅ 顯示關鍵幀資訊
    print("\n🎯 關鍵幀偵測結果：")
    print(f"🟢 出手點 Frame:   {release_frame}")
    print(f"🟢 踏地點 Frame:   {landing_frame}")
    print(f"🟢 肩膀最開 Frame: {shoulder_frame}")

    # ✅ 步驟 4：執行特徵擷取
    print("\n🧾 踏地特徵分析：")
    landing_features = detect_landing_features(pose_sequence, landing_frame)
    for key, value in landing_features.items():
        print(f"  {key}: {value}")

    print("\n🧾 肩膀特徵分析：")
    shoulder_features = extract_shoulder_features(
        pose_sequence, shoulder_frame, landing_frame
    )
    for key, value in shoulder_features.items():
        print(f"  {key}: {value}")

    print("\n🧾 出手特徵分析：")
    release_features = extract_release_features(pose_sequence, release_frame)
    for key, value in release_features.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
