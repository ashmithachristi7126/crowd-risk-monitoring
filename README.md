# 🚦 Crowd Risk Monitoring System

> Real-time crowd safety analysis using deep learning — moving beyond counting to understanding behavior.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-red?style=flat-square)
![OpenCV](https://img.shields.io/badge/CV-OpenCV-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## 📌 Overview

Large public gatherings don't become dangerous because of size alone — but because of **how crowds behave**.

This project presents a real-time crowd risk monitoring system that goes beyond simple head counting. It intelligently analyzes crowd **density**, **movement patterns**, **spatial distribution**, and **behavioral anomalies** to classify crowd conditions into actionable risk levels:

| Risk Level | Indicator | Description |
|---|---|---|
| Low | 🟢 | Crowd is sparse and well-distributed |
| Medium | 🟡 | Density is rising; monitoring recommended |
| High | 🔴 | Dangerous density or abnormal behavior detected |

> ⚠️ **Important:** Model weights (`.pt` files) and datasets are **not included** due to size constraints. Download a pretrained YOLOv8 model from [Ultralytics](https://github.com/ultralytics/ultralytics) or train your own using `train.py` and `data.yaml` before running the system.

---

## ⚙️ Key Features

| Feature | Description |
|---|---|
| 🔍 Real-Time Person Detection | YOLO-based detection for identifying individuals in each frame |
| 🎯 Multi-Object Tracking | ByteTrack-inspired tracking for behavioral understanding across frames |
| 📊 Hybrid Crowd Counting | Combines detection-based and density-based estimation to reduce undercounting in dense scenes |
| 📈 Density Estimation | Normalized crowd density calculation using spatial coverage metrics |
| 🚨 Risk Classification Engine | Rule-based system that outputs Low / Medium / High risk levels |
| 🎥 Flexible Input Support | Works with both live video streams and pre-recorded footage |

---

## 🧠 How It Works

The system follows a structured processing pipeline:

```
Input (Video / Live Stream)
        │
        ▼
Frame Preprocessing & Resizing
        │
        ▼
Person Detection  ──────────────────────────  YOLOv8
        │
        ▼
Multi-Object Tracking  ─────────────────────  ByteTrack concept
        │
        ▼
Crowd Density Calculation
        │
        ▼
Hybrid Crowd Count Estimation
  ├── Detection-based count
  └── Density-based estimation (handles occlusion)
        │
        ▼
Risk Classification  ───────────  🟢 Low  /  🟡 Medium  /  🔴 High
        │
        ▼
Visual Output + Alerts
  ├── Bounding boxes per person
  ├── Crowd count overlay
  ├── Density value
  └── Risk level banner
```

> Detection alone is unreliable in dense scenes. The hybrid counting strategy combines detection and density estimation to significantly improve accuracy in crowded environments.

---

## 📂 Project Structure

```
crowd-risk-monitoring/
│
├── main.py            # Entry point — runs the full pipeline
├── detect.py          # YOLOv8-based person detection
├── track.py           # Multi-object tracking logic
├── density.py         # Crowd density calculation
├── anomaly.py         # Risk classification and anomaly detection
├── train.py           # Model training script
├── main_video.py      # Video file processing pipeline
├── data.yaml          # Dataset configuration
├── requirements.txt   # Python dependencies
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip
- A pretrained YOLOv8 `.pt` model ([download here](https://github.com/ultralytics/ultralytics))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/crowd-risk-monitoring.git
cd crowd-risk-monitoring

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your YOLOv8 model weights in the project root
#    e.g., yolov8n.pt or your custom trained model

# 4. Run the system
python main.py
```

### Training a Custom Model

If you want to train on your own dataset:

```bash
python train.py --data data.yaml --epochs 50
```

---

## 📊 Output

For each processed frame, the system produces:

- **Bounding boxes** drawn around each detected person
- **Crowd count** — total persons in frame
- **Density value** — normalized spatial density score
- **Risk level** — color-coded Low / Medium / High banner

This combination of visual and numerical output makes the system practical for real-world deployment in control rooms and monitoring dashboards.

---

## 🧪 Core Technologies

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Multi-Object Tracking | ByteTrack-inspired approach |
| Density Estimation | Spatial coverage normalization |
| Counting Strategy | Hybrid (detection + density) |
| Risk Classification | Rule-based engine |
| Computer Vision | OpenCV |

---

## 🎯 Applications

- Smart city surveillance infrastructure
- Railway stations, airports, and transit hubs
- Stadiums, concerts, and large public events
- Crowd disaster prevention and emergency response

---

## 🔮 Future Improvements

-  Adaptive risk thresholds based on venue type and event context
-  Advanced motion behavior analysis (stampede, bottleneck detection)
-  Larger and more diverse dataset training
-  Real-time alert system integration (SMS, webhook, dashboard)
-  Edge deployment support (Jetson, Raspberry Pi)

---

> *"This project moves from counting people to understanding crowd risk — which is where real safety begins."*
