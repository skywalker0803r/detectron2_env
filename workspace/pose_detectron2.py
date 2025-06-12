# pose_detector.py
import os
import cv2
import numpy as np
import torch
from tqdm.notebook import tqdm
import supervision as sv

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog


def run_detectron2(video_path, output_dir, device,save_img=False):
    """
    使用 Detectron2 對影片執行骨架偵測，並輸出 keypoints .npy 與繪圖圖片。
    - video_path: 輸入影片路徑
    - output_dir: 儲存結果的資料夾
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Detectron2 初始化 ---
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.DEVICE = device

    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    tracker = sv.ByteTrack(frame_rate=30)

    # --- 開始處理影片 ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ 無法開啟影片：{video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker.frame_rate = fps

    print(f"🎞️ 處理影片：{video_path}，總影格：{frame_count}，FPS: {fps}")

    target_person_track_id = -1

    for frame_idx in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        outputs = predictor(rgb)

        xyxy = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        confidence = outputs["instances"].scores.cpu().numpy()
        class_id = (
            outputs["instances"].pred_classes.cpu().numpy()
            if outputs["instances"].has("pred_classes")
            else np.zeros(len(xyxy), dtype=int)
        )

        key_points_data = None
        if outputs["instances"].has("pred_keypoints"):
            kp = outputs["instances"].pred_keypoints.cpu().numpy()
            key_points_data = kp[:, :, :2]  # (N, 17, 2)

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            data={"key_points": key_points_data} if key_points_data is not None else {},
        )

        detections = tracker.update_with_detections(detections)

        current_target = None
        if len(detections.tracker_id) > 0:
            if target_person_track_id == -1:
                target_person_track_id = detections.tracker_id[0]
                print(
                    f"👤 Frame {frame_idx}: 設定目標人物 ID = {target_person_track_id}"
                )

            target_indices = np.where(detections.tracker_id == target_person_track_id)[
                0
            ]
            if len(target_indices) > 0:
                idx = target_indices[0]
                current_target = {
                    "xyxy": detections.xyxy[idx],
                    "confidence": detections.confidence[idx],
                    "class_id": detections.class_id[idx],
                    "key_points": (
                        detections.data["key_points"][idx]
                        if "key_points" in detections.data
                        else None
                    ),
                }

        # 畫圖 & 儲存
        display_frame = frame.copy()
        if current_target and current_target["key_points"] is not None:
            image_h, image_w = frame.shape[:2]
            d2_instances = type(outputs["instances"])((image_h, image_w))
            d2_instances.pred_boxes = outputs["instances"].pred_boxes.__class__(
                np.expand_dims(current_target["xyxy"], axis=0)
            )
            d2_instances.scores = torch.tensor(
                [float(current_target["confidence"])], dtype=torch.float32
            )

            kp_xy = current_target["key_points"]
            kp_conf = np.full((1, kp_xy.shape[0], 1), 2.0)
            kp_full = np.concatenate((kp_xy[None, :, :], kp_conf), axis=2)
            d2_instances.pred_keypoints = torch.tensor(kp_full, dtype=torch.float32)

            v = Visualizer(
                display_frame[:, :, ::-1],
                metadata=metadata,
                scale=1.0,
                instance_mode=ColorMode.IMAGE,
            )
            vis = v.draw_instance_predictions(d2_instances.to("cpu"))
            display_frame = vis.get_image()[:, :, ::-1]

            np.save(
                os.path.join(output_dir, f"frame_{frame_idx:05d}_target_keypoints.npy"),
                kp_full[0],
            )

        if save_img == True:
            cv2.imwrite(
                os.path.join(output_dir, f"frame_{frame_idx:05d}_annotated.jpg"),
                display_frame,
            )

    cap.release()
    if save_img == True:
        print("✅ Detectron2 處理完畢，骨架資料與圖片已儲存。")
    else:
        print("✅ Detectron2 處理完畢，骨架資料已儲存。")


# ✅ 可獨立執行測試
if __name__ == "__main__":
    VIDEO_PATH = "data/Yu_Darvish_FF_videos_4S/pitch_0001.mp4"
    OUTPUT_DIR = "output_detectron2_first_person_tracked"
    run_detectron2(VIDEO_PATH, OUTPUT_DIR)