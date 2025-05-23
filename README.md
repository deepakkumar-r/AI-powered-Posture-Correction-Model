# 🧍‍♂️ Slouchometer – AI-Powered Posture Correction System

**Slouchometer** is a real-time posture correction system built using **MediaPipe** and **OpenCV**, designed to help users maintain good sitting posture and reduce health issues caused by prolonged slouching.

---

## 🧠 Project Overview

Slouchometer is a lightweight, Python-based tool that:

- Tracks user posture using a webcam
- Uses **MediaPipe Pose Estimation** to detect body landmarks
- Calculates back and neck angles to monitor slouching
- Displays a **gentle popup alert** when poor posture is detected
- Includes a **cooldown timer** to avoid repetitive alerting

Perfect for students, remote workers, gamers, or anyone spending long hours in front of a screen!

---

## ⚙️ Features

- 🧍‍♀️ Real-time body tracking via webcam
- 📐 Posture angle calculation (neck & back)
- 🔔 Slouch detection with gentle desktop alerts
- ⏱️ Cooldown mechanism to avoid alert spam
- 💻 Lightweight and easy to deploy on laptops/desktops

---

## 🛠️ Tech Stack

- Python 3
- [MediaPipe](https://developers.google.com/mediapipe) – for human pose detection
- [OpenCV](https://opencv.org/) – for webcam access and image processing
- Tkinter – for alert popup UI

---

## 🚀 Getting Started

Follow these steps to set up and run the Slouchometer locally:

### 1. Clone this repository

```bash
git clone https://github.com/deepakkumar-r/AI-powered-Posture-Correction-Model.git
cd AI-powered-Posture-Correction-Model


2. Install Required Packages
pip install -r requirements.txt
# or manually
pip install mediapipe opencv-python


3.Run the Application:
python slouchometer.py


Folder Structure
AI-powered-Posture-Correction-Model/
│
├── slouchometer.py         # Main posture detection script
├── requirements.txt        # Required Python packages
├── README.md               # Project documentation
└── assets/ (optional)      # UI images or sample outputs



 How It Works
Camera Access: OpenCV accesses your webcam in real-time.

Pose Detection: MediaPipe identifies key posture landmarks like shoulders and hips.

Angle Analysis: Calculates the back/neck angle using coordinates.

Slouch Detection: If the angle crosses a threshold, an alert popup appears.

Cooldown Timer: Alerts are limited to avoid spamming the user.




Deepak Kumar
Pre-Final-year AIML Undergraduate
📫 Email: deepakkumarvijayin@gmail.com
🌐 GitHub: @deepakkumar-r

