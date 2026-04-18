import os
import cv2
from density import calculate_density
from anomaly import detect_risk_level
from detect import detect_people_tiled  # ✅ added for tiled detection
from density import estimate_count_from_coverage  # ✅ added for estimated count

IMAGE_DIR = "test_images"

def draw_boxes(frame, detections):
    for x1, y1, x2, y2, score in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{score:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def main():
    if not os.path.isdir(IMAGE_DIR):
        print(f"❌ Folder not found: {IMAGE_DIR}")
        return

    images = sorted([f for f in os.listdir(IMAGE_DIR)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    if not images:
        print(f"❌ No images found in {IMAGE_DIR}")
        return

    for idx, img_name in enumerate(images, start=1):
        img_path = os.path.join(IMAGE_DIR, img_name)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"⚠️ Skipping unreadable: {img_name}")
            continue

        h, w = frame.shape[:2]

        # ✅ replaced with tiled detection (only change requested)
        detections = detect_people_tiled(frame, conf=0.10, imgsz=960, tiles=2, iou_nms=0.5)


        # ✅ replaced count calculation with hybrid (only change requested)
        frame_area = w * h

        yolo_count = len(detections)

        est_count, coverage = estimate_count_from_coverage(detections, frame_area)

        # if YOLO count seems too low in dense scenes, use estimated count
        if yolo_count < 50 and coverage > 0.10:
            count = est_count
        else:
            count = yolo_count

        density_box = coverage
        risk = detect_risk_level(count, density_box)

        draw_boxes(frame, detections)

        cv2.putText(frame, f"Image: {idx}/{len(images)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Count: {count}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Density: {density_box:.3f}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(frame, f"Risk: {risk}", (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Crowd Risk Monitoring (SPACE=next, ESC=exit)", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
