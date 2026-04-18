import os
import cv2
from detect import detect_people_tiled
from density import calculate_density
from anomaly import detect_risk_level

VIDEO_DIR = "videos"

def draw_boxes(frame, detections):
    for x1, y1, x2, y2, score in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Optional (comment if slow)
        # cv2.putText(frame, f"{score:.2f}", (x1, max(15, y1 - 5)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def main():
    if not os.path.isdir(VIDEO_DIR):
        print(f"❌ Video folder not found: {VIDEO_DIR}")
        return

    videos = sorted([
        f for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ])

    if not videos:
        print(f"❌ No videos found in {VIDEO_DIR}")
        return

    print(f"✅ Found {len(videos)} videos. Playing one by one...")

    for v_index, v_name in enumerate(videos, start=1):
        video_path = os.path.join(VIDEO_DIR, v_name)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️ Skipping (cannot open): {v_name}")
            continue

        print(f"\n▶ Playing video {v_index}/{len(videos)}: {v_name}")

        frame_id = 0

        # ✅ NEW: keep last results so skipped frames still show boxes
        last_detections = []
        last_count = 0
        last_density_box = 0.0
        last_risk = "NORMAL"

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            # ✅ Resize for speed (keep as you had)
            frame = cv2.resize(frame, (960, 540))
            h, w = frame.shape[:2]

            # ✅ Run YOLO only every 5th frame
            if frame_id % 5 == 0:
                detections = detect_people_tiled(frame, conf=0.15, imgsz=640, tiles=2, iou_nms=0.5)

                count, density_box = calculate_density(detections, w, h)
                risk = detect_risk_level(count, density_box)

                # ✅ store results
                last_detections = detections
                last_count = count
                last_density_box = density_box
                last_risk = risk

            # ✅ ALWAYS draw last detections (even on skipped frames)
            draw_boxes(frame, last_detections)

            cv2.putText(frame, f"Video: {v_index}/{len(videos)}  {v_name}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, f"Count: {last_count}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, f"Density: {last_density_box:.3f}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
            cv2.putText(frame, f"Risk: {last_risk}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            cv2.imshow("Crowd Risk Monitoring (ESC=next video, Q=quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC = next video
                break
            if key == ord("q") or key == ord("Q"):
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()

    cv2.destroyAllWindows()
    print("\n✅ Finished all videos.")

if __name__ == "__main__":
    main()

