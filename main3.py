import cv2
import dlib
import pyautogui
import numpy as np
from imutils import face_utils

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Get the indexes for the eye regions from the facial landmark model
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def get_gaze_ratio(eye, frame):
    # Create mask for the eye region (ensure mask is grayscale and the same size as the input frame)
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [eye], 255)

    # Extract the eye region from the grayscale frame
    eye_region = cv2.bitwise_and(frame, frame, mask=mask)

    # Get bounding box of the eye region
    min_x = np.min(eye[:, 0])
    max_x = np.max(eye[:, 0])
    min_y = np.min(eye[:, 1])
    max_y = np.max(eye[:, 1])

    # Crop the eye from the frame
    eye_img = eye_region[min_y:max_y, min_x:max_x]

    # Apply a threshold to isolate the pupil
    _, threshold_eye = cv2.threshold(eye_img, 70, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded eye to locate the pupil
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Get the largest contour which is likely the pupil
        contour = max(contours, key=lambda c: cv2.contourArea(c))
        M = cv2.moments(contour)
        if M['m00'] > 0:
            # Calculate center of the pupil
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy), eye_img.shape[1], eye_img.shape[0]
    
    return None, eye_img.shape[1], eye_img.shape[0]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEyeCenter = midpoint(leftEye[0], leftEye[3])
        rightEyeCenter = midpoint(rightEye[0], rightEye[3])

        # Extract gaze ratios for both eyes
        left_gaze, left_eye_w, left_eye_h = get_gaze_ratio(leftEye, gray)  # Use gray frame
        right_gaze, right_eye_w, right_eye_h = get_gaze_ratio(rightEye, gray)

        if left_gaze and right_gaze:
            # Gaze direction based on pupil position relative to eye width/height
            left_ratio_x = (left_gaze[0] - left_eye_w / 2) / (left_eye_w / 2)
            left_ratio_y = (left_gaze[1] - left_eye_h / 2) / (left_eye_h / 2)
            right_ratio_x = (right_gaze[0] - right_eye_w / 2) / (right_eye_w / 2)
            right_ratio_y = (right_gaze[1] - right_eye_h / 2) / (right_eye_h / 2)

            # Average ratios from both eyes to determine overall gaze direction
            gaze_x = (left_ratio_x + right_ratio_x) / 2
            gaze_y = (left_ratio_y + right_ratio_y) / 2

            # Move mouse relative to gaze direction
            screen_width, screen_height = pyautogui.size()
            move_x = screen_width / 2 + (screen_width / 2) * gaze_x
            move_y = screen_height / 2 + (screen_height / 2) * gaze_y

            pyautogui.moveTo(move_x, move_y)

        # Display the eyes with detected pupils
        cv2.polylines(frame, [leftEye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [rightEye], True, (0, 255, 0), 1)
        if left_gaze:
            cv2.circle(frame, (leftEye[0][0] + left_gaze[0], leftEye[0][1] + left_gaze[1]), 2, (0, 0, 255), -1)
        if right_gaze:
            cv2.circle(frame, (rightEye[0][0] + right_gaze[0], rightEye[0][1] + right_gaze[1]), 2, (0, 0, 255), -1)

    # Show the frame
    cv2.imshow("Eye Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
