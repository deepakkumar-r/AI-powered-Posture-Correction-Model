import cv2
import numpy as np
import mediapipe as mp
import pygame
import pandas as pd
import datetime
import os
from openpyxl import load_workbook
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_spine_angle(landmarks):
    """
    Calculate the spine angle using shoulder and hip midpoints.
    """
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Calculate midpoints
    shoulder_mid = np.array([(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2])
    hip_mid = np.array([(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2])

    # Calculate spine vector (from hip midpoint to shoulder midpoint)
    spine_vector = shoulder_mid - hip_mid
    vertical_vector = np.array([0, -1])  # Vertical line (upwards)

    # Calculate the angle between spine vector and vertical axis
    angle = np.degrees(np.arccos(np.dot(spine_vector, vertical_vector) / (np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector))))
    return angle

def calculate_head_position_change(landmarks, initial_head_mid):
    """
    Calculate the vertical change in the head position.
    """
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]

    # Calculate head midpoint
    head_mid = np.array([(left_ear.x + right_ear.x) / 2, (left_ear.y + right_ear.y) / 2])

    # Calculate vertical change in the head position
    vertical_change = initial_head_mid[1] - head_mid[1]
    return vertical_change

def calculate_shoulder_breadth_change(landmarks, initial_shoulder_breadth):
    """
    Calculate the change in shoulder breadth.
    """
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Calculate current shoulder breadth
    current_shoulder_breadth = abs(left_shoulder.x - right_shoulder.x)

    # Calculate change in shoulder breadth
    breadth_change = current_shoulder_breadth - initial_shoulder_breadth
    return breadth_change

def play_beep():
    """
    Play a beep sound to alert the user.
    """
    pygame.mixer.init()
    pygame.mixer.music.load("slouchbeep.mp3")  # or use .wav
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    
def get_posture_status(is_slouching_spine, is_slouching_head, is_slouching_shoulders):
    """
    Determine the overall posture status based on the slouching indicators.
    """
    if not any([is_slouching_spine, is_slouching_head, is_slouching_shoulders]):
        return "Good Posture"
    
    slouching_parts = []
    if is_slouching_spine:
        slouching_parts.append("Spine")
    if is_slouching_head:
        slouching_parts.append("Head")
    if is_slouching_shoulders:
        slouching_parts.append("Shoulders")
    
    return "Slouching: " + ", ".join(slouching_parts)

def initialize_excel():
    """
    Initialize the Excel file for data storage.
    Returns the file path.
    """
    # Create directory for data if it doesn't exist
    data_dir = "posture_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Create file path with current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    excel_path = os.path.join(data_dir, f"posture_data.xlsx")
    
    # Check if file exists, create if it doesn't
    if not os.path.exists(excel_path):
        # Create a DataFrame with the necessary columns
        columns = [
            "timestamp", 
            "posture_status",  # New comprehensive status column
            "spine_angle", 
            "head_position_change", 
            "shoulder_breadth_change", 
            "is_slouching_spine", 
            "is_slouching_head", 
            "is_slouching_shoulders",
            "session_id", 
            "user_id"
        ]
        df = pd.DataFrame(columns=columns)
        df.to_excel(excel_path, index=False)
        print(f"Created new Excel file at {excel_path}")
    else:
        print(f"Using existing Excel file at {excel_path}")
    
    return excel_path

def add_data_to_excel(excel_path, data_row):
    """
    Add a new row of data to the Excel file.
    """
    try:
        # Read the existing data
        df = pd.read_excel(excel_path)
        
        # Append the new data row
        df = pd.concat([df, pd.DataFrame([data_row])], ignore_index=True)
        
        # Write back to Excel
        df.to_excel(excel_path, index=False)
        return True
    except Exception as e:
        print(f"Error updating Excel file: {e}")
        return False

def main():
    # Generate a unique session ID for this monitoring session
    session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Set user ID (could be made configurable)
    user_id = "user_1"
    
    # Initialize Excel file
    excel_path = initialize_excel()
    
    # Data collection interval in seconds (collect data every 5 seconds)
    data_interval = 5
    last_data_time = time.time()
    
    # Open webcam
    cap = cv2.VideoCapture(0)

    # Calibration phase
    print("Calibrating... Please sit in a neutral upright posture.")
    calibration_frames = 100  # Number of frames to use for calibration
    calibration_spine_angles = []
    calibration_head_positions = []
    calibration_shoulder_breadths = []

    while len(calibration_spine_angles) < calibration_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Get landmarks
            landmarks = results.pose_landmarks.landmark

            # Calculate spine angle
            spine_angle = calculate_spine_angle(landmarks)
            calibration_spine_angles.append(spine_angle)

            # Calculate head position
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
            head_mid = np.array([(left_ear.x + right_ear.x) / 2, (left_ear.y + right_ear.y) / 2])
            calibration_head_positions.append(head_mid)

            # Calculate shoulder breadth
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            shoulder_breadth = abs(left_shoulder.x - right_shoulder.x)
            calibration_shoulder_breadths.append(shoulder_breadth)

        # Show frame
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate thresholds
    if calibration_spine_angles and calibration_head_positions and calibration_shoulder_breadths:
        spine_angle_threshold = np.mean(calibration_spine_angles) + 10  # Add buffer for slouching
        initial_head_mid = np.mean(calibration_head_positions, axis=0)  # Average head position
        initial_shoulder_breadth = np.mean(calibration_shoulder_breadths)  # Average shoulder breadth
        head_position_threshold = -0.04  # Define a threshold for head position change
        shoulder_breadth_threshold = -0.04  # Threshold for significant shoulder breadth narrowing
        print(f"Calibration complete. Spine angle threshold: {spine_angle_threshold:.2f}°, Head position threshold: {head_position_threshold:.2f}, Shoulder breadth threshold: {shoulder_breadth_threshold:.2f}")

    # Main loop for posture detection
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Get landmarks
            landmarks = results.pose_landmarks.landmark

            # Calculate spine angle
            spine_angle = calculate_spine_angle(landmarks)

            # Calculate head position change
            head_position_change = calculate_head_position_change(landmarks, initial_head_mid)

            # Calculate shoulder breadth change
            shoulder_breadth_change = calculate_shoulder_breadth_change(landmarks, initial_shoulder_breadth)

            # Check for slouching based on spine angle
            is_slouching_spine = spine_angle > spine_angle_threshold
            if is_slouching_spine:
                cv2.putText(frame, "Slouching: Spine", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                play_beep()

            # Check for slouching based on head position
            is_slouching_head = head_position_change < head_position_threshold
            if is_slouching_head:
                cv2.putText(frame, "Slouching: Head", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                play_beep()

            # Check for slouching based on shoulder breadth
            is_slouching_shoulders = shoulder_breadth_change < shoulder_breadth_threshold
            if is_slouching_shoulders:
                cv2.putText(frame, "Slouching: Shoulders", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                play_beep()

            # Get overall posture status
            posture_status = get_posture_status(is_slouching_spine, is_slouching_head, is_slouching_shoulders)
            
            # Display posture status on the frame
            status_color = (0, 255, 0) if posture_status == "Good Posture" else (0, 0, 255)
            cv2.putText(frame, f"Status: {posture_status}", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            # Display information on the screen
            cv2.putText(frame, f"Spine Angle: {spine_angle:.2f}°", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Head Position Change: {head_position_change:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Shoulder Breadth Change: {shoulder_breadth_change:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Data collection every few seconds
            current_time = time.time()
            if current_time - last_data_time >= data_interval:
                # Create data row
                data_row = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "posture_status": posture_status,  # Add comprehensive posture status
                    "spine_angle": round(spine_angle, 2),
                    "head_position_change": round(head_position_change, 4),
                    "shoulder_breadth_change": round(shoulder_breadth_change, 4),
                    "is_slouching_spine": int(is_slouching_spine),  # 1 for True, 0 for False
                    "is_slouching_head": int(is_slouching_head),
                    "is_slouching_shoulders": int(is_slouching_shoulders),
                    "session_id": session_id,
                    "user_id": user_id
                }
                
                # Add data to Excel
                success = add_data_to_excel(excel_path, data_row)
                if success:
                    cv2.putText(frame, "Data recorded", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Error recording data", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Update last data collection time
                last_data_time = current_time

            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show frame
        cv2.imshow("Posture Correction", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"Session completed. Data saved to {excel_path}")
    print(f"This file can now be used for data analysis in Power BI.")
        
if __name__ == "__main__":
    main()