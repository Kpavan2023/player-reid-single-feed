# 🏃‍♂️ Player Re-Identification in Sports Footage (Single-Camera)

## 📌 Project Summary
This project detects and consistently re-identifies players in a sports video captured by a single camera using:
- **YOLOv8 (Ultralytics)**: For real-time player detection.
- **StrongSORT**: For tracking and assigning persistent player IDs.
- **Custom-trained ReID model (osnet_x0_25)**: Learned player appearance features using the **DukeMTMC-ReID** dataset.

This allows tracking players across frames even with motion blur or occlusion.

---

## 🧠 Methodology
- **Detection**: A YOLOv8 model (`best2.pt`) trained specifically to detect players in sports footage.
- **ReID Training**: Used ResNet18 backbone on DukeMTMC-ReID dataset. Model saved as `osnet_x0_25.pt`.
- **Tracking**: StrongSORT integrates motion and appearance features for robust multi-player tracking.
- **Output**: Generates `annotated_output.mp4` with unique ID tags over each player bounding box.

---

## 🗂️ Folder Structure
```
player-reid-single-feed/
├── option2_single_feed/
│   ├── best2.pt                  # YOLOv8 model for player detection
│   ├── player_tracker.py         # Main script for player tracking
│   ├── 15sec_input_720p.mp4      # Input video for evaluation
│   └── output/
│       └── annotated_output.mp4  # Output video with annotated tracking results
│
├── train_reid/
│    └──train_reid_model.ipynb      # Colab or Jupyter notebook used for training
│    
├── strongsort/
│   ├── strongsort.py             # StrongSORT tracker logic using boxmot
│   └── osnet_x0_25.pt            # Trained ReID model (appearance embedding)
│
├── config/
│   └── strongsort.yaml           # Tracker configuration file
│
├── Player_Reid_Report.pdf        # Final report explaining approach and findings
├── requirements.txt              # Python dependencies for environment setup
└── README.md                     # Setup instructions and project overview
```

---

## ⚙️ Setup & Run

### 1. Install Dependencies
```bash
git clone <your_repo_url>
cd player_reid
python -m venv venv
source venv/bin/activate        # For Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Tracker
```bash
cd option2_single_feed
python player_tracker.py
```

✅ Annotated video saved to: `output/annotated_output.mp4`

---

## 📦 Dependencies
```txt
ultralytics
opencv-python
torch
torchvision
matplotlib
git+https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet.git
```

---

## 📄 Report
See `player_reid_report.pdf` for:
- Training details
- Techniques and experiments
- Challenges and improvements

---
---

### 👨‍💻 Developed By

**Pavan Kumar K**

---
## 📜 Licensing / Credits
- **StrongSORT** and **boxmot** (by Mikel Broström) used under AGPL-3.0.
- **YOLOv8** by Ultralytics.
- This repo is for academic evaluation use only.
