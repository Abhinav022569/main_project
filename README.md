# 🖐️ Air Mouse v3: Gesture Control System

A cutting-edge Python-based desktop control system that leverages **computer vision** (MediaPipe) to track hand gestures in real-time. It enables users to fully control the mouse cursor, perform clicks, and manage windows using natural hand movements, offering a hands-free alternative to traditional input devices.

---

## 💻 Tech Stack

| Component | Technology | Requirement/Role |
| :--- | :--- | :--- |
| **Primary Language**| Python 3.x | Core logic and scripting. |
| **Hand Tracking** | MediaPipe (Google) | High-accuracy, real-time landmark detection. |
| **Desktop Control** | PyAutoGUI | Simulating mouse movement and input. |
| **Camera Interface**| OpenCV (`cv2`) | Capturing and processing video stream. |
| **Utility** | NumPy | Advanced array operations for robust camera health checks. |

---

## ✨ Core Features

### 🖱️ Mouse Control & Interaction

* **Cursor Movement:** Uses a **Pointing** gesture (Index finger extended) with built-in **Relative Movement** and **Smoothing** to translate hand motion into precise screen position.
* **Click/Drag:** Performed by **Pinching** the thumb and index finger together (pinch distance compared to a `CLICK_THRESHOLD`). This supports continuous drag-and-drop operations.
* **Control Activation:** A preventative, two-step gesture sequence (**Thumbs Up** followed by an **Open Palm**) is required to toggle mouse control *on* or *off*, avoiding accidental activation.

### 🖥️ System & Utility

* **Window Resize:** Activated by simultaneously forming a **Dual-Fist** gesture with both hands. Moving the fists apart or together controls the resizing of the active window via operating system commands (Alt+Space+S).
* **Camera Fallback:** Features a robust camera health check with **automatic fallback** from an external camera (Index 1) to the laptop camera (Index 0) if the primary video feed is blank or static.

---

## 🚀 Installation & Setup

### Prerequisites

| Component | Minimum Requirement |
| :--- | :--- |
| **CPU** | Pentium dual-core or above |
| **RAM** | 2 GB or above |
| **OS** | WINDOWS 7 or higher / Linux |

### Quick Setup

The implementation requires a Python environment with the following libraries:

1.  **Install Required Libraries:**
    Run the following command to install all necessary dependencies:
    ```bash
    pip install opencv-python mediapipe pyautogui numpy
    ```

2.  **Run the System:**
    Execute the main control script to start tracking:
    ```bash
    python air_mousev2.py
    ```
    *Note: The script will automatically attempt to locate a healthy webcam and may take a moment to initialize.*

## 🔒 Default Credentials (Usage Notes)

| Input Method | Gesture/Action | State |
| :--- | :--- | :--- |
| **Activation Toggle** | Thumbs Up, then Open Palm | Switches control **ON/OFF** |
| **Mouse Pointer** | Index Finger Extended | Cursor **Movement** |
| **Click/Hold** | Pinch Thumb & Index | Left **Click** or **Drag** |
