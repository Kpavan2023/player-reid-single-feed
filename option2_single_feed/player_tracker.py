import os
import cv2
import sys
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# Add strongsort to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from strongsort.strongsort import StrongSort

# ---------- CONFIGURATION ----------
VIDEO_PATH = "15sec_input_720p.mp4"
YOLO_MODEL_PATH = "best2.pt"  # Trained YOLOv8 player detector
REID_MODEL_PATH = Path(__file__).resolve().parent.parent / 'strongsort' / 'osnet_x0_25.pt'

OUTPUT_DIR = "output"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "annotated_output.mp4")

CONF_THRESH = 0.25
IOU_THRESH = 0.45
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- LOAD MODELS ----------
yolo_model = YOLO(YOLO_MODEL_PATH)
tracker = StrongSort(
    reid_weights=REID_MODEL_PATH,
    device=DEVICE,
    half=False,
    max_iou_dist=0.7,
    max_age=30,
    n_init=3,
    nn_budget=100
)

# ---------- VIDEO READING ----------
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# ---------- PROCESS VIDEO ----------
frame_idx = 0
with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.predict(frame, conf=CONF_THRESH, iou=IOU_THRESH, device=DEVICE, verbose=False)
        dets = results[0].boxes

        if dets is None or dets.shape[0] == 0:
            out.write(frame)
            frame_idx += 1
            pbar.update(1)
            continue

        xyxy = dets.xyxy.cpu().numpy()
        confs = dets.conf.cpu().numpy()
        clss = dets.cls.cpu().numpy()

        detections = []
        for bbox, conf, cls in zip(xyxy, confs, clss):
            if int(cls) == 0:  # Only track players
                x1, y1, x2, y2 = bbox
                detections.append([x1, y1, x2, y2, conf, cls])

        detections = np.array(detections)

        if len(detections) > 0:
            tracks = tracker.update(dets=detections, img=frame)

            for track in tracks:
                x1, y1, x2, y2, track_id, conf, cls, _ = track.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Player {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        out.write(frame)
        frame_idx += 1
        pbar.update(1)

cap.release()
out.release()
print(f"\n✅ Done! Annotated video saved at: {OUTPUT_PATH}")
