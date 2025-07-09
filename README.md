# Player_re-identification_liat-ai
This project methodology aims to identify each player in a sports footage and ensure that players who go out of frame and reappear are assigned the same identity as before.

![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

# Player Re-Identification in Sports Footage using YOLOv11 custom model.

This project demonstrates player tracking and re-identification across video frames using object detection and simple tracking logic. It ensures consistent ID assignment for each player, even if they go out of view and reappear.

##  Folder Structure

```bash
Player_Re-Identification/
├── input/
│ └── 15sec_input_720p.mp4       # Input video
├── utils/
│ └── iou.py                     # code that re-identifies players over time using simple re-ID logic        
├── weights/
│ └── best.pt                    # YOLOv11 custom model
├── output/
│ └── tracked_video.mp4          # Output video with ID labels
├── reid_tracker.py              # Main script
├── README.md                    # This file
├── requirements.txt             # all modules and dependencies required to install   
├── _report.pdf                # Final brief report
```

All required materials are available here:
https://drive.google.com/drive/folders/1Nx6H_n0UUI6L-6i8WknXd4Cv2c3VjZTP

##  Setup Instructions

### 1.  Clone or download this repo
```bash
git clone https://github.com/Samartha21BRS1698/Player_re-identification_liat-ai.git
cd Player_ReIdentification
```

### 2. Create and activate virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Run the Code

Make sure: weights/best.pt (YOLOv11 custom model) is present
link:- https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view

input/15sec_input_720p.mp4 is the target video

Then run:
python reid_tracker.py

The output will be saved in:
output/tracked_video.mp4


## Features
-> Uses YOLOv11 fine-tuned for players (class ID = 2)
-> Assigns consistent unique IDs to each player
-> Tracks players even after they disappear and re-enter
-> Handles multiple re-entries, occlusions, and partial overlaps
-> Capped ID range to prevent jumping (IDs range from 0–29)

## Technologies Used
-> Python 3.10+ — Core programming language used
-> YOLOv11 (Ultralytics) — Fine-tuned object detection model for players and ball
-> OpenCV — Frame-by-frame video processing, drawing bounding boxes and saving output
-> NumPy — For numerical operations and bounding box comparisons
-> Virtual Environment (venv) — Environment isolation to manage dependencies

## Limitations
-> No deep appearance-based tracking (yet)
-> Players with identical appearance may swap IDs occasionally
-> Currently optimized for one camera feed only

## Future Work
-> Integrating DeepSORT or color histogram appearance features
-> Extend to multi-camera cross-feed Re-ID
-> Adding re-ID heatmaps and tracking analytics

## Author

**Samartha**  
B.Tech student 
🎓 AI/ML • Data Science •  Computer Vision • Google Cloud 
🔗 [LinkedIn](https://www.linkedin.com/in/samartha-b0154a293) | [GitHub](https://github.com/Samartha21BRS1698)

## 📝 License
 MIT License © 2025 Samartha