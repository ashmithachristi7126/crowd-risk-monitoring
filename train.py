from ultralytics import YOLO
import os

def train_model():
    os.makedirs("models", exist_ok=True)

    # If you really have YOLOv12 weights, use them.
    # Otherwise use yolov8n.pt (works with same API).
    weights = "yolov12.pt"
    if not os.path.exists(weights):
        weights = "yolov8n.pt"

    model = YOLO(weights)

    model.train(
        data="data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        name="crowd_yolo"
    )

    # Best weights path created by Ultralytics:
    # runs/detect/crowd_yolo/weights/best.pt
    best_pt = os.path.join("runs", "detect", "crowd_yolo", "weights", "best.pt")
    if os.path.exists(best_pt):
        os.replace(best_pt, os.path.join("models", "crowd_yolov12.pt"))
        print("✅ Saved trained model to models/crowd_yolov12.pt")
    else:
        print("⚠️ Training finished, but best.pt not found. Check runs/detect/ folder.")

if __name__ == "__main__":
    train_model()
