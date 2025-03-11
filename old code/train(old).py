"""
Training script for the face anti-spoofing model using YOLOv8.
"""

from ultralytics import YOLO

def main():
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    # Train the model using the dataset
    model.train(
        data='../Dataset/SplitData/data.yaml',
        epochs=3,
        imgsz=640,
        batch=16,
        name='face_antispoofing'
    )

if __name__ == '__main__':
    main() 