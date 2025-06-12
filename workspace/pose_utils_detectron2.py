import os
import cv2
import numpy as np


def load_pose_sequence(keypoint_dir):
    """
    讀取 Detectron2 輸出的 .npy 檔，回傳包含 frame index 和 keypoints 的序列。
    每筆格式為：
    {
        "frame": int,
        "keypoints": np.ndarray (17, 3)  # [x, y, score]
    }
    """
    sequence = []
    files = sorted(
        [f for f in os.listdir(keypoint_dir) if f.endswith("_target_keypoints.npy")]
    )

    for f in files:
        frame_idx = int(f.split("_")[1])
        keypoints = np.load(os.path.join(keypoint_dir, f))  # shape = (17, 3)
        sequence.append({"frame": frame_idx, "keypoints": keypoints})
    return sequence


def calculate_pixel_angle_from_points(a, b, c):
    """
    使用三個 numpy pixel 座標點 a, b, c 計算夾角（degree）。
    - a, b, c 為 shape = (2,) 的 numpy 陣列
    - b 為頂點
    """
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle


def get_pose_window(pose_sequence, center_frame, window_size):
    """
    取得前後 window_size 幀內的骨架資料。
    回傳 list，每筆為 {frame, keypoints}
    """
    center_idx = None
    for i, item in enumerate(pose_sequence):
        if item["frame"] == center_frame:
            center_idx = i
            break
    if center_idx is None:
        return []

    window = []
    for offset in range(-window_size, window_size + 1):
        idx = center_idx + offset
        if 0 <= idx < len(pose_sequence):
            window.append(pose_sequence[idx])
    return window


def visualize_pose_features(
    window_items,
    image_dir,
    save_dir="output_vis",
    save_prefix="feature",
    connections=[],
    highlight_points=[],
    text=None,
    extra_info_per_frame=None,
    min_score=0.3,
):
    """
    將 keypoints 標記到原始圖片上，支援連線、重點畫圈與文字註記。
    - window_items: [{frame, keypoints}]
    - image_dir: 原始輸出圖片資料夾（frame_xxxxx_annotated.jpg）
    - connections: [(idx1, idx2)] 關節連線
    - highlight_points: [idx] 要特別標註的關節點
    - text: 傳入函式 (info) -> str，用於顯示角度等文字
    - extra_info_per_frame: list，每幀對應一筆 info 資料傳入 text()
    - min_score: 關鍵點可信度閾值
    """
    os.makedirs(save_dir, exist_ok=True)

    for i, item in enumerate(window_items):
        frame_idx = item["frame"]
        keypoints = item["keypoints"]  # shape = (17, 3)
        image_path = os.path.join(image_dir, f"frame_{frame_idx:05d}_annotated.jpg")
        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        def pt(i):
            return tuple(map(int, keypoints[i, :2]))

        # 畫連線
        for a, b in connections:
            if keypoints[a, 2] > min_score and keypoints[b, 2] > min_score:
                cv2.line(image, pt(a), pt(b), (0, 255, 0), 2)

        # 畫重點
        for p in highlight_points:
            if keypoints[p, 2] > min_score:
                cv2.circle(image, pt(p), 5, (0, 0, 255), -1)

        # 顯示文字
        if text:
            label = text(extra_info_per_frame[i] if extra_info_per_frame else {})
            if highlight_points:
                cv2.putText(
                    image,
                    label,
                    pt(highlight_points[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        save_path = os.path.join(save_dir, f"{save_prefix}_{frame_idx}.jpg")
        cv2.imwrite(save_path, image)
