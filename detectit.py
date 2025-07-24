## SET UR ID TELEGRAM BEFORE USE THIS ##
import cv2
import numpy as np
import os
import requests
from ultralytics import YOLO
import supervision as sv  
from concurrent.futures import ThreadPoolExecutor
import time    
import threading
import pickle

# Telegram Bot Configuration
ID_CONTACT = 'YOUR-CONTACT-ID-TO-RECEIVE-MESSAGE'  
TOKEN = 'YOUR-TOKEN' #Get from @BotFather 
TELEGRAM_URL = f'https://api.telegram.org/bot{TOKEN}/sendPhoto'


# Load YOLOv8 model
model = YOLO("yolov8s_new.pt")  # Path to your trained model

# Class Names (update based on your training labels)
CLASS_NAMES = ["helm", "motor", "nohelmet", "plat"]

# Ask user for video input type
video_input = input("Enter 'file' for video file or 'webcam' for webcam: ").strip().lower()

if video_input == 'file':
    video_path = input("Enter the path to your video file: ").strip()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        exit()
    # ByteTrack initialization for video file
    video_info = sv.VideoInfo.from_video_path(video_path=video_path)
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)
elif video_input == 'webcam':
    # Webcam input option (default 0 or other options like 1, 2, 3)
    camera_id = input("Enter webcam ID (0 for default, 1 for external webcam 1, 2 for external webcam 2, etc.): ").strip()

    # Set default to 0 if no valid input is provided
    if camera_id == '' or not camera_id.isdigit():
        camera_id = 0
    else:
        camera_id = int(camera_id)

    # Open the chosen webcam
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Cannot open webcam {camera_id}.")
        exit()

    byte_track = sv.ByteTrack(frame_rate=30)  
else:
    print("Invalid input. Exiting.")
    exit()

# Allow manual resize of window
cv2.namedWindow("Helmet Detection", cv2.WINDOW_NORMAL)

# Tracker status
motor_tracker = {}
no_helmet_tracker = {}
processed_logs = set()

# Load or initialize counter from pickle
counter_file = "counter.pkl"

if os.path.exists(counter_file):
    with open(counter_file, 'rb') as f:
        counter = pickle.load(f)
else:
    counter = 1

# Thread Pool for Parallel Uploading
executor = ThreadPoolExecutor(max_workers=5)
file_lock = threading.Lock()

def send_to_telegram(image_path, caption):
    """Function to send image to Telegram."""
    try:
        with open(image_path, 'rb') as file:
            files = {'photo': file}
            data = {'chat_id': ID_CONTACT, 'caption': caption}
            response = requests.post(TELEGRAM_URL, files=files, data=data)

        if response.status_code == 200:
            print(f"✅ Sent to Telegram: {caption}")
        elif response.status_code == 429:
            wait_time = response.json().get('parameters', {}).get('retry_after', 5)
            print(f"⚠️ Too many requests. Waiting {wait_time} seconds...")
            time.sleep(wait_time)        
        else:
            print(f"❌ Failed to send: {response.json()}")
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)  # Safely remove file here
            print(f"🗑️ File deleted: {image_path}")        

pkl_file_path = counter_file  

# Detection Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break
    
    video_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / (video_info.fps if video_input == 'file' else 30)  
    formatted_time = f"{video_time:012.2f}"

    # YOLOv8 Inference
    results = model(frame, verbose=False, conf=0.4, iou=0.4) #set the thresholds and performs object detection
    result = results[0]
    detections = result.boxes

    if detections is not None and len(detections) > 0:
        detections_sv = sv.Detections.from_ultralytics(result)
        tracked_detections = byte_track.update_with_detections(detections_sv)

        for xyxy, class_id, tracker_id in zip(
            tracked_detections.xyxy, tracked_detections.class_id, tracked_detections.tracker_id
        ):
            class_id = int(class_id)
            x1, y1, x2, y2 = map(int, xyxy)

            # Draw bounding boxes and labels on the video
            label = f"{CLASS_NAMES[class_id]} ID:{tracker_id} (x:{x1}, y:{y1})"
            # Draw bounding box and labels
            if class_id == 0:  # Helm
                color = (0, 255, 0)  # Green
            elif class_id == 1:  # Motor
                color = (255, 255, 0)  # Cyan
            elif class_id == 2:  # No Helmet
                color = (0, 0, 255)  # Red
            elif class_id == 3:  # Plat
                color = (255, 255, 255)  # White            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Track motor and no-helmet
            if class_id == 1:  # Motor detected
                motor_tracker[tracker_id] = (x1, y1, x2, y2)

                # Check for no helmet and plat number inside motor box
                no_helmet_detected = False
                plate_coords = None

                for sub_xyxy, sub_class_id, _ in zip(
                    tracked_detections.xyxy, tracked_detections.class_id, tracked_detections.tracker_id
                ):
                    sx1, sy1, sx2, sy2 = map(int, sub_xyxy)
                    if sub_class_id == 2 and x1 <= sx1 <= x2 and y1 <= sy1 <= y2:
                        no_helmet_detected = True
                    elif sub_class_id == 3 and x1 <= sx1 <= x2 and y1 <= sy1 <= y2:
                        plate_coords = (sx1, sy1, sx2, sy2)

                if no_helmet_detected and plate_coords:
                    mx1, my1, mx2, my2 = motor_tracker[tracker_id]
                    px1, py1, px2, py2 = plate_coords

                    # Crop motor and plate images
                    motor_img = frame[my1:my2, mx1:mx2]
                    plate_img = frame[py1:py2, px1:px2]

                    # Combine images
                    combined_img = np.zeros((max(motor_img.shape[0], plate_img.shape[0]), motor_img.shape[1] + plate_img.shape[1], 3), dtype=np.uint8)
                    combined_img[:motor_img.shape[0], :motor_img.shape[1]] = motor_img
                    combined_img[:plate_img.shape[0], motor_img.shape[1]:] = plate_img

                    caption_text = f"{counter}. Motor {tracker_id} tanpa helm dengan plat nomor {formatted_time} (xy: {x1}, {y1})"
                    temp_filename = f"{counter}_motor_{tracker_id}_plat_{formatted_time}.jpg"

                    # Save and send
                    cv2.putText(combined_img, caption_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.imwrite(temp_filename, combined_img)
                    executor.submit(send_to_telegram, temp_filename, caption_text)

                    print(f"✅{counter}. Motor {tracker_id} tanpa helm dengan plat nomor terdeteksi.")
                    # Increment counter and save to pickle
                    counter += 1
                    with open(counter_file, 'wb') as f:
                        pickle.dump(counter, f)                

    # Display the video frame
    resized_frame = cv2.resize(frame, (800, 600))
    cv2.imshow("Helmet Detection", resized_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
executor.shutdown(True)

if os.path.exists(pkl_file_path):
    os.remove(pkl_file_path)
