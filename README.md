# 🍟 McDonald's Study Guardian

A small **satirical computer-vision side project** that makes it harder to get distracted while studying.

The concept is simple:

> If you look away from your computer, McDonald's appears. 🍟

## 🎯 How It Works

McDonald's Study Guardian uses your webcam and facial pose estimation to determine whether you're looking at your computer.

- 👨‍💻 **Looking at the computer** → Study normally
- 📱 **Looking away / looking down** → McDonald's webpage appears and a distraction sound plays
- 👀 **Looking back at the computer** → The webpage closes and the sound stops
- 🔄 **Look away again** → It triggers again

The system is designed to react quickly without requiring a long countdown or timer.

## 🧠 Technology

- **Python**
- **OpenCV** — webcam capture and video processing
- **MediaPipe Face Landmarker** — facial landmark detection and head-pose estimation
- **Pygame** — audio playback
- **Chrome automation** — opening and closing the distraction webpage
- **Facial pose estimation** — detecting changes in pitch and yaw
- 
🚀 Running the Project
1. Clone the repository
git clone https://github.com/omarayyyman07/mcdonalds-study-guardian.git
cd mcdonalds-study-guardian

2. Create a virtual environment
python -m venv .venv

Activate it on Windows:

.venv\Scripts\Activate.ps1

3. Install dependencies
pip install opencv-python mediapipe pygame numpy
4. Run
python main.py

Make sure your webcam is available and that the required project files are in the same directory.

🍟 Why McDonald's?

Because a serious productivity application would have been boring.

This project started as a silly idea and turned into a quick experiment with computer vision, head-pose estimation, automation, and real-time interaction.

⚠️ Disclaimer

This is a satirical and purely educational side project.

It is not affiliated with, sponsored by, or endorsed by McDonald's.

The project was created for fun and experimentation and was intentionally kept small rather than being developed as a serious commercial product.

Made for studying.

Unfortunately, McDonald's has other plans. 🍟

## 📁 Project Structure

```text
McDonald's Study Guardian/
│
├── main.py
├── face_landmarker.task
├── distraction.mp3
├── mcdonalds.html
├── mcdonalds_files/
└── README.md

