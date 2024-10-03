# Eye Tracking with Gaze Control

## Overview
This project implements an eye-tracking system that allows users to control the mouse cursor using their gaze direction. The system uses the `dlib` library for facial landmark detection and `pyautogui` for mouse control. By analyzing the position of the pupils within the eye regions, the system determines the user's gaze and moves the cursor accordingly.

## Features
- Real-time eye tracking using webcam input.
- Detection of pupil position to determine gaze direction.
- Mouse cursor control based on gaze movement.
- Visual feedback with bounding boxes around the eyes and detected pupils.

## Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.x
- pip (Python package installer)

## Installation
To set up the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd eye-tracking
    ```

2. Install required Python packages:
    ```bash
    pip install opencv-python dlib pyautogui imutils numpy
    ```

3. Download the facial landmark predictor model:
   - You will need the `shape_predictor_68_face_landmarks.dat` file from [dlib's model files](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2). Decompress the file and place it in your project directory.

## Usage
1. **Run the Eye Tracking Script**:
   Use the following command to start the application:
   ```bash
   python eye_tracking.py
