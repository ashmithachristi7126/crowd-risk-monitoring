from ultralytics import YOLO
import numpy as np

# Stronger model for crowded scenes
model = YOLO("yolov8l.pt")


def nms(boxes, scores, iou_thresh=0.5):
    """Simple NMS to remove duplicate boxes."""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep

def detect_people_tiled(frame, conf=0.05, imgsz=1280, tiles=3, iou_nms=0.5):
    """
    Detect people using tiling (works better for dense crowds).
    tiles=3 means 3x3 grid.
    """
    H, W = frame.shape[:2]

    tile_h = H // tiles
    tile_w = W // tiles

    all_boxes = []
    all_scores = []

    for r in range(tiles):
        for c in range(tiles):
            y1 = r * tile_h
            x1 = c * tile_w

            y2 = H if r == tiles - 1 else (r + 1) * tile_h
            x2 = W if c == tiles - 1 else (c + 1) * tile_w

            tile = frame[y1:y2, x1:x2]

            results = model.predict(tile, conf=conf, imgsz=imgsz, iou=0.6, verbose=False)

            for res in results:
                if res.boxes is None:
                    continue
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id != 0:  # COCO person
                        continue

                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    score = float(box.conf[0])

                    # map back to original image coords
                    bx1 += x1
                    bx2 += x1
                    by1 += y1
                    by2 += y1

                    all_boxes.append([bx1, by1, bx2, by2])
                    all_scores.append(score)

    # apply NMS over all tiles
    keep_idx = nms(all_boxes, all_scores, iou_thresh=iou_nms)

    detections = []
    for i in keep_idx:
        x1, y1, x2, y2 = all_boxes[i]
        detections.append([int(x1), int(y1), int(x2), int(y2), float(all_scores[i])])

    return detections
