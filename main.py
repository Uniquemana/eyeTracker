import cv2
import dlib
import pyautogui
import numpy as np
from imutils import face_utils

# Initialize dlib's face detector (HOG-based) and create facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from dlib's model zoo

# Get the indexes for the eye regions from the facial landmark model
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Function to calculate eye aspect ratio (EAR) to detect blink
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        # Determine facial landmarks for the face region
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate the center of each eye
        leftEyeCenter = leftEye.mean(axis=0).astype("int")
        rightEyeCenter = rightEye.mean(axis=0).astype("int")

        # Draw circles at the eye centers
        cv2.circle(frame, tuple(leftEyeCenter), 3, (0, 255, 0), -1)
        cv2.circle(frame, tuple(rightEyeCenter), 3, (0, 255, 0), -1)

        # Calculate the direction to move the mouse
        eyeCenter = ((leftEyeCenter + rightEyeCenter) / 2).astype("int")
        screenWidth, screenHeight = pyautogui.size()

        # Map the center of the eyes to the screen resolution
        screenX = np.interp(eyeCenter[0], [0, frame.shape[1]], [0, screenWidth])
        screenY = np.interp(eyeCenter[1], [0, frame.shape[0]], [0, screenHeight])

        # Move the mouse cursor
        pyautogui.moveTo(screenX, screenY)

    # Show the frame with eye centers
    cv2.imshow("Eye Tracking", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

