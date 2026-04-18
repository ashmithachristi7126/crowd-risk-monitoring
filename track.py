from ultralytics import YOLO
import os

MODEL_PATH = "models/crowd_yolov12.pt"

if os.path.exists(MODEL_PATH):
    tracker = YOLO(MODEL_PATH)
else:
    try:
        tracker = YOLO("yolov12.pt")
    except Exception:
        tracker = YOLO("yolov8n.pt")

def track_people(frame, conf=0.4):
    """
    For VIDEO streams: returns list of [x1, y1, x2, y2, track_id, score]
    For IMAGES: IDs may be None or unstable.
    """
    results = tracker.track(frame, persist=True, conf=conf, verbose=False)
    tracks = []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            score = float(box.conf[0])

            track_id = -1
            if box.id is not None:
                track_id = int(box.id[0])

            tracks.append([x1, y1, x2, y2, track_id, score])

    return tracks
