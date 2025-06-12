import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import supervision as sv  # 用於追蹤

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# --- 設定輸入與輸出資料夾 ---
VIDEO_PATH = "data/Yu_Darvish_FF_videos_4S/pitch_0001.mp4"
OUTPUT_DIR = "output_detectron2_first_person_tracked"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Detectron2 設定 ---
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
)
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

# --- 處理影片 ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ 無法打開影片：{VIDEO_PATH}")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    print("⚠️ 無法取得正確 FPS，預設使用 30")
    fps = 30

tracker = sv.ByteTrack(frame_rate=int(round(fps)))
target_person_track_id = -1

print(f"處理影片：{VIDEO_PATH}")
print(f"總影格數：{frame_count}")
print(f"幀率 (FPS)：{fps}")

for frame_idx in tqdm(range(frame_count)):
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    outputs = predictor(image_rgb)

    xyxy = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    confidence = outputs["instances"].scores.cpu().numpy()
    class_id = (
        outputs["instances"].pred_classes.cpu().numpy()
        if outputs["instances"].has("pred_classes")
        else np.zeros(len(outputs["instances"]), dtype=int)
    )

    key_points_data = None
    if outputs["instances"].has("pred_keypoints"):
        key_points_data = outputs["instances"].pred_keypoints.cpu().numpy()

    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        data={"key_points": key_points_data} if key_points_data is not None else {},
    )

    detections = tracker.update_with_detections(detections)

    current_target_person_data = None
    if len(detections.tracker_id) > 0:
        if target_person_track_id == -1:
            target_person_track_id = detections.tracker_id[0]
            print(
                f"🎬 影格 {frame_idx}: 首次識別到目標人物ID為 {target_person_track_id}"
            )
            current_target_person_data = {
                "xyxy": detections.xyxy[0],
                "confidence": detections.confidence[0],
                "class_id": detections.class_id[0],
                "key_points": (
                    detections.data["key_points"][0]
                    if "key_points" in detections.data
                    and detections.data["key_points"] is not None
                    else None
                ),
            }
        else:
            target_indices = np.where(detections.tracker_id == target_person_track_id)[
                0
            ]
            if len(target_indices) > 0:
                target_idx = target_indices[0]
                current_target_person_data = {
                    "xyxy": detections.xyxy[target_idx],
                    "confidence": detections.confidence[target_idx],
                    "class_id": detections.class_id[target_idx],
                    "key_points": (
                        detections.data["key_points"][target_idx]
                        if "key_points" in detections.data
                        and detections.data["key_points"] is not None
                        else None
                    ),
                }
            else:
                print(
                    f"🚨 影格 {frame_idx}: 目標人物ID {target_person_track_id} 丟失。"
                )

    display_frame = frame.copy()
    if current_target_person_data is not None:
        image_h, image_w, _ = frame.shape
        d2_instances = type(outputs["instances"])((image_h, image_w))

        target_xyxy = current_target_person_data["xyxy"]
        if target_xyxy.ndim == 1:
            target_xyxy = np.expand_dims(target_xyxy, axis=0)
        d2_instances.pred_boxes = outputs["instances"].pred_boxes.__class__(target_xyxy)

        target_confidence = current_target_person_data["confidence"]
        if isinstance(target_confidence, np.ndarray) and target_confidence.ndim == 0:
            score_value = target_confidence.item()
        elif isinstance(target_confidence, (float, np.float32, np.float64)):
            score_value = float(target_confidence)
        else:
            score_value = (
                target_confidence[0].item() if target_confidence.size > 0 else 0.0
            )
        d2_instances.scores = torch.tensor([score_value], dtype=torch.float32)

        if current_target_person_data["key_points"] is not None:
            kp_xy = current_target_person_data["key_points"]
            if kp_xy.ndim == 2:
                kp_xy = np.expand_dims(kp_xy, axis=0)
            if kp_xy.shape[2] == 2:
                kp_conf = np.full((kp_xy.shape[0], kp_xy.shape[1], 1), 2.0)
                kp_xy = np.concatenate((kp_xy, kp_conf), axis=2)
            d2_keypoints_tensor = torch.tensor(kp_xy, dtype=torch.float32)
            d2_instances.pred_keypoints = outputs["instances"].pred_keypoints.__class__(
                d2_keypoints_tensor
            )

        v = Visualizer(
            display_frame[:, :, ::-1],
            metadata=metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE,
        )
        vis = v.draw_instance_predictions(d2_instances.to("cpu"))
        display_frame = vis.get_image()[:, :, ::-1]

        if d2_instances.has("pred_keypoints"):
            target_keypoints_to_save = d2_instances.pred_keypoints[0].cpu().numpy()
            np.save(
                os.path.join(OUTPUT_DIR, f"frame_{frame_idx:05d}_target_keypoints.npy"),
                target_keypoints_to_save,
            )

    out_path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:05d}_annotated.jpg")
    cv2.imwrite(out_path, display_frame)

cap.release()
cv2.destroyAllWindows()
print("影片處理完成，所有影格圖片已保存。")
