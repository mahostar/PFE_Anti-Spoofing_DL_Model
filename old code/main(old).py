"""
Main script for real-time face anti-spoofing detection.
"""

import math
import time
import cv2
import cvzone
from ultralytics import YOLO

def main():
    confidence = 0.6
    
    # Initialize camera
    cap = cv2.VideoCapture(1)  # For Webcam
    cap.set(3, 640)
    cap.set(4, 480)
    
    # Load model
    model = YOLO("../models/l_version_1_300.pt")
    classNames = ["fake", "real"]
    
    # FPS calculation
    prev_frame_time = 0
    new_frame_time = 0
    
    while True:
        new_frame_time = time.time()
        success, img = cap.read()
        results = model(img, stream=True, verbose=False)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                
                if conf > confidence:
                    color = (0, 255, 0) if classNames[cls] == 'real' else (0, 0, 255)
                    
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                    cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
                                    (max(0, x1), max(35, y1)), scale=2, thickness=4,
                                    colorR=color, colorB=color)
        
        # Calculate and display FPS
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print(f"FPS: {fps:.2f}")
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main() 