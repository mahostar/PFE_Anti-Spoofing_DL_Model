"""
DepthFusion-ViT Data Collection Tool
-----------------------------------
This script provides a GUI application for collecting face data to train 
an anti-spoofing model. It detects faces in real-time video and saves 
labeled images with separate controls for real and fake data collection.
"""

import cv2
import os
import time
import numpy as np
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
from datetime import datetime
import urllib.parse
import torch
import torchvision
import torch.nn as nn
from collections import deque
import socket

# Optional import of MTCNN - will fall back to OpenCV if not available
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False


class CameraThread(threading.Thread):
    """Dedicated thread for camera reading to maximize frame rate"""
    def __init__(self, camera_url, frame_buffer, stop_event, timeout=10.0):
        super().__init__()
        self.daemon = True
        self.camera_url = camera_url
        self.frame_buffer = frame_buffer
        self.stop_event = stop_event
        self.timeout = timeout
        self.fps = 0
        
    def run(self):
        # Set timeout for socket operations
        socket.setdefaulttimeout(self.timeout)
        
        # Configure capture parameters for better performance
        cap = cv2.VideoCapture(self.camera_url, cv2.CAP_FFMPEG)
        
        # MJPEG optimizations
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Small buffer for low latency
        
        if not cap.isOpened():
            print(f"Failed to open camera: {self.camera_url}")
            return
            
        frame_times = deque(maxlen=30)
        last_frame_time = time.time()
        
        try:
            while not self.stop_event.is_set():
                start_time = time.time()
                
                ret, frame = cap.read()
                
                if not ret:
                    print("Camera read failed, reconnecting...")
                    time.sleep(0.5)
                    # Recreate capture
                    cap.release()
                    cap = cv2.VideoCapture(self.camera_url, cv2.CAP_FFMPEG)
                    continue
                
                # Calculate FPS
                current_time = time.time()
                frame_times.append(current_time)
                
                # Update FPS every 5 frames
                if len(frame_times) > 5:
                    time_diff = frame_times[-1] - frame_times[0]
                    if time_diff > 0:
                        self.fps = len(frame_times) / time_diff
                
                # If the buffer is full, remove oldest frame
                if len(self.frame_buffer) >= self.frame_buffer.maxlen:
                    self.frame_buffer.popleft()
                    
                # Add new frame to buffer
                self.frame_buffer.append(frame)
                
                # Throttle if needed to avoid excessive CPU usage
                elapsed = time.time() - start_time
                if elapsed < 0.01:  # Target up to 100 FPS for capture
                    time.sleep(0.01 - elapsed)
                    
        except Exception as e:
            print(f"Camera thread error: {str(e)}")
        finally:
            # Clean up
            if cap and cap.isOpened():
                cap.release()
            print("Camera thread stopped")


class FaceDataCollectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DepthFusion-ViT Data Collection Tool")
        self.root.geometry("1200x720")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Check CUDA availability
        self.cuda_available = False
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.cuda_available = self.device.type == 'cuda'
            print(f"Using device: {self.device}")
            if self.cuda_available:
                print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"GPU detection error: {str(e)}")
            self.device = torch.device('cpu')
        
        # Variables
        self.save_location = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "DepthFusionViT_Data"))
        self.is_recording = False
        self.is_collecting_real = False
        self.is_collecting_fake = False
        self.camera_source = tk.StringVar(value="0")
        self.camera_url = tk.StringVar(value="http://192.168.1.13:8080/video")
        self.real_count = 0
        self.fake_count = 0
        self.frame_count = 0
        self.processed_count = 0
        self.blur_threshold = tk.IntVar(value=50)
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.detector_type = tk.StringVar(value="OpenCV")  # Default to OpenCV as it's more reliable
        self.fps_display = tk.StringVar(value="Camera: 0 FPS | Processing: 0 FPS")
        self.resolution = tk.StringVar(value="640x480")
        self.detect_faces = tk.BooleanVar(value=True)
        self.process_every_n_frames = tk.IntVar(value=2)  # Process every n frames
        
        # Frame buffer for the camera thread
        self.frame_buffer = deque(maxlen=10)
        
        # Camera thread
        self.camera_thread = None
        
        # Initialize face detector
        self.initialize_detector()
        
        # Initialize video capture
        self.cap = None
        
        # Create GUI components
        self.create_widgets()
        
        # Add GPU info to status
        if self.cuda_available:
            gpu_info = f"Using GPU: {torch.cuda.get_device_name(0)}"
        else:
            gpu_info = "Using CPU (CUDA not available or not working)"
        
        # Initialization message
        messagebox.showinfo("Welcome", 
                          f"Welcome to DepthFusion-ViT Data Collection Tool\n\n"
                          f"{gpu_info}\n\n"
                          "1. Set your save location\n"
                          "2. Configure camera (local or IP webcam)\n"
                          "3. Start the camera\n"
                          "4. Use 'Collect Real' or 'Collect Fake' buttons\n"
                          "5. Position faces in the frame\n\n"
                          "Images are automatically saved when faces are detected.")
    
    def initialize_detector(self):
        """Initialize the appropriate face detector"""
        if self.detector_type.get() == "MTCNN" and MTCNN_AVAILABLE:
            try:
                self.face_detector = MTCNN(
                    keep_all=True,
                    device=self.device,
                    select_largest=False,
                    min_face_size=80,
                    thresholds=[0.6, 0.7, 0.7],  # More lenient thresholds
                    post_process=False
                )
                print("Using MTCNN face detector")
            except Exception as e:
                print(f"Error initializing MTCNN: {str(e)}")
                self.detector_type.set("OpenCV")
                self.face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                print("Falling back to OpenCV face detector")
        else:
            # OpenCV's detector
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("Using OpenCV face detector")

    def create_widgets(self):
        # Main frame layout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Camera selection
        camera_frame = ttk.LabelFrame(control_frame, text="Camera Settings")
        camera_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Camera type selection
        type_frame = ttk.Frame(camera_frame)
        type_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(type_frame, text="Camera Type:").pack(side=tk.LEFT)
        self.camera_type = ttk.Combobox(type_frame, values=["Local Camera", "IP Webcam"])
        self.camera_type.set("IP Webcam")
        self.camera_type.pack(side=tk.LEFT, padx=5)
        self.camera_type.bind('<<ComboboxSelected>>', self.on_camera_type_change)
        
        # Local camera selection
        self.local_frame = ttk.Frame(camera_frame)
        ttk.Label(self.local_frame, text="Camera ID:").pack(side=tk.LEFT)
        camera_dropdown = ttk.Combobox(self.local_frame, textvariable=self.camera_source, 
                                     values=["0", "1", "2", "3"])
        camera_dropdown.pack(side=tk.LEFT, padx=5)
        
        # IP webcam URL
        self.ip_frame = ttk.Frame(camera_frame)
        self.ip_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(self.ip_frame, text="IP Webcam URL:").pack(side=tk.LEFT)
        ttk.Entry(self.ip_frame, textvariable=self.camera_url, width=40).pack(side=tk.LEFT, padx=5)
        
        # Resolution selection
        res_frame = ttk.Frame(camera_frame)
        res_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(res_frame, text="Resolution:").pack(side=tk.LEFT)
        res_dropdown = ttk.Combobox(res_frame, textvariable=self.resolution, 
                                  values=["320x240", "640x480", "800x600", "1280x720"])
        res_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Performance options
        perf_frame = ttk.Frame(camera_frame)
        perf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Face detection toggle
        ttk.Checkbutton(perf_frame, text="Enable Face Detection", 
                      variable=self.detect_faces).pack(side=tk.LEFT, padx=5)
        
        # Frame processing rate
        proc_frame = ttk.Frame(camera_frame)
        proc_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(proc_frame, text="Process every N frames:").pack(side=tk.LEFT)
        ttk.Spinbox(proc_frame, from_=1, to=10, textvariable=self.process_every_n_frames, 
                  width=5).pack(side=tk.LEFT, padx=5)
                  
        # Face detector selection
        detector_frame = ttk.Frame(camera_frame)
        detector_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(detector_frame, text="Face Detector:").pack(side=tk.LEFT)
        detector_dropdown = ttk.Combobox(detector_frame, textvariable=self.detector_type, 
                                       values=["OpenCV", "MTCNN"])
        detector_dropdown.pack(side=tk.LEFT, padx=5)
        detector_dropdown.bind('<<ComboboxSelected>>', lambda e: self.initialize_detector())
        
        # Save location
        location_frame = ttk.LabelFrame(control_frame, text="Save Location")
        location_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Entry(location_frame, textvariable=self.save_location, width=30).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(location_frame, text="Browse", command=self.browse_location).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Camera control
        camera_control_frame = ttk.LabelFrame(control_frame, text="Camera Control")
        camera_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.camera_button = ttk.Button(camera_control_frame, text="Start Camera", command=self.toggle_camera)
        self.camera_button.pack(padx=5, pady=5, fill=tk.X)
        
        # Blur threshold
        blur_frame = ttk.Frame(camera_control_frame)
        blur_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(blur_frame, text="Blur Threshold:").pack(side=tk.LEFT)
        blur_scale = ttk.Scale(blur_frame, from_=10, to=100, variable=self.blur_threshold)
        blur_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(blur_frame, textvariable=tk.StringVar(value=lambda: str(self.blur_threshold.get()))).pack(side=tk.LEFT, padx=5)
        
        # Collection controls
        collection_frame = ttk.LabelFrame(control_frame, text="Data Collection")
        collection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Real data collection
        real_frame = ttk.Frame(collection_frame)
        real_frame.pack(fill=tk.X, padx=5, pady=5)
        self.real_button = ttk.Button(real_frame, text="Collect Real", command=self.toggle_real_collection)
        self.real_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.real_label = ttk.Label(real_frame, text="Real: 0")
        self.real_label.pack(side=tk.LEFT, padx=5)
        
        # Fake data collection
        fake_frame = ttk.Frame(collection_frame)
        fake_frame.pack(fill=tk.X, padx=5, pady=5)
        self.fake_button = ttk.Button(fake_frame, text="Collect Fake", command=self.toggle_fake_collection)
        self.fake_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.fake_label = ttk.Label(fake_frame, text="Fake: 0")
        self.fake_label.pack(side=tk.LEFT, padx=5)
        
        # Status frame
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(status_frame, textvariable=self.fps_display).pack(side=tk.RIGHT, padx=5)
        
        # Right panel - Video display
        display_frame = ttk.LabelFrame(main_frame, text="Camera Feed")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(display_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
    
    def browse_location(self):
        """Open file dialog to select save location"""
        folder = filedialog.askdirectory(initialdir=self.save_location.get())
        if folder:
            self.save_location.set(folder)
            
            # Create subdirectories
            for subdir in ["real", "fake"]:
                os.makedirs(os.path.join(folder, subdir), exist_ok=True)
            
            self.status_var.set(f"Save location set to: {folder}")
    
    def toggle_camera(self):
        """Start or stop the camera"""
        if not self.is_recording:
            self.start_camera()
            self.camera_button.config(text="Stop Camera")
        else:
            self.stop_camera()
            self.camera_button.config(text="Start Camera")
    
    def toggle_real_collection(self):
        """Toggle real data collection"""
        if not self.is_recording:
            messagebox.showinfo("Info", "Please start the camera first")
            return
        
        self.is_collecting_real = not self.is_collecting_real
        self.is_collecting_fake = False  # Stop fake collection if active
        
        if self.is_collecting_real:
            self.real_button.config(text="Stop Real")
            self.fake_button.config(state=tk.DISABLED)
            self.status_var.set("Collecting REAL samples...")
        else:
            self.real_button.config(text="Collect Real")
            self.fake_button.config(state=tk.NORMAL)
            self.status_var.set("Stopped collecting REAL samples")
    
    def toggle_fake_collection(self):
        """Toggle fake data collection"""
        if not self.is_recording:
            messagebox.showinfo("Info", "Please start the camera first")
            return
        
        self.is_collecting_fake = not self.is_collecting_fake
        self.is_collecting_real = False  # Stop real collection if active
        
        if self.is_collecting_fake:
            self.fake_button.config(text="Stop Fake")
            self.real_button.config(state=tk.DISABLED)
            self.status_var.set("Collecting FAKE samples...")
        else:
            self.fake_button.config(text="Collect Fake")
            self.real_button.config(state=tk.NORMAL)
            self.status_var.set("Stopped collecting FAKE samples")
    
    def on_camera_type_change(self, event=None):
        """Handle camera type selection change"""
        if self.camera_type.get() == "Local Camera":
            self.local_frame.pack(fill=tk.X, padx=5, pady=5)
            self.ip_frame.pack_forget()
        else:
            self.local_frame.pack_forget()
            self.ip_frame.pack(fill=tk.X, padx=5, pady=5)
    
    def start_camera(self):
        """Start the camera and face detection"""
        try:
            self.stop_event.clear()
            
            # Clear buffer
            self.frame_buffer.clear()
            
            # Get camera URL or device
            if self.camera_type.get() == "Local Camera":
                camera_src = int(self.camera_source.get())
            else:
                url = self.camera_url.get()
                if not url.endswith('/video'):
                    url = url.rstrip('/') + '/video'
                print(f"Connecting to IP webcam at: {url}")
                camera_src = url
            
            # Start camera thread for better performance
            self.camera_thread = CameraThread(
                camera_src, 
                self.frame_buffer,
                self.stop_event
            )
            self.camera_thread.start()
            
            # Start processing
            self.is_recording = True
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self.process_camera)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            # Start display thread
            self.display_thread = threading.Thread(target=self.update_display)
            self.display_thread.daemon = True
            self.display_thread.start()
            
            self.status_var.set("Camera started")
            
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
            self.is_recording = False
    
    def stop_camera(self):
        """Stop the camera and processing"""
        if self.is_recording:
            self.stop_event.set()
            self.is_recording = False
            self.is_collecting_real = False
            self.is_collecting_fake = False
            
            # Wait for threads to finish
            if hasattr(self, 'processing_thread'):
                self.processing_thread.join(timeout=1.0)
            
            if hasattr(self, 'display_thread'):
                self.display_thread.join(timeout=1.0)
            
            # Camera thread will terminate automatically
            
            self.status_var.set("Camera stopped")
            
            # Clear canvas
            self.canvas.delete("all")
            
            # Update button text
            self.camera_button.config(text="Start Camera")
            self.real_button.config(text="Collect Real")
            self.fake_button.config(text="Collect Fake")
    
    def process_camera(self):
        """Process frames and detect faces"""
        # For FPS calculation
        frame_times = deque(maxlen=30)
        
        while not self.stop_event.is_set():
            try:
                # Skip if no frames available
                if not self.frame_buffer:
                    time.sleep(0.01)  # Short sleep to avoid CPU spinning
                    continue
                
                # Get the latest frame
                frame = self.frame_buffer[-1].copy()
                
                # Skip frames if needed for performance
                self.frame_count += 1
                if self.frame_count % self.process_every_n_frames.get() != 0:
                    # Still display frame but skip detection
                    if not self.frame_queue.full():
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                        self.frame_queue.put(rgb_frame)
                    continue
                
                self.processed_count += 1
                
                # Update FPS calculation
                current_time = time.time()
                frame_times.append(current_time)
                
                # Keep only recent frames for FPS calculation
                if len(frame_times) > 1:
                    fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                    if self.camera_thread:
                        camera_fps = self.camera_thread.fps
                    else:
                        camera_fps = 0
                    # Update FPS display every 10 frames
                    if self.processed_count % 10 == 0:
                        self.fps_display.set(f"Camera: {camera_fps:.1f} FPS | Processing: {fps:.1f} FPS")
                
                # Skip face detection if disabled (for performance)
                faces = []
                if self.detect_faces.get():
                    # Resize for faster processing if needed
                    h, w = frame.shape[:2]
                    process_frame = frame
                    
                    # Downsample large frames for faster processing
                    if max(h, w) > 800:
                        scale = 800 / max(h, w)
                        process_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                    
                    # Detect faces using the selected detector
                    if self.detector_type.get() == "MTCNN" and MTCNN_AVAILABLE:
                        try:
                            # Convert to RGB for PyTorch
                            frame_rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                            
                            # Convert to PyTorch tensor
                            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()
                            frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension
                            
                            # Move to GPU if available
                            if self.cuda_available:
                                frame_tensor = frame_tensor.to(self.device)
                            
                            # Detect faces
                            boxes, _ = self.face_detector.detect(frame_tensor)
                            
                            if boxes is not None and len(boxes[0]) > 0:
                                boxes = boxes[0]  # Get boxes from first (and only) image in batch
                                
                                # Rescale coordinates if frame was resized
                                if process_frame is not frame:
                                    scale_x = w / process_frame.shape[1]
                                    scale_y = h / process_frame.shape[0]
                                    for i in range(len(boxes)):
                                        # Scale back to original frame size
                                        x, y, x2, y2 = [int(b) for b in boxes[i]]
                                        x = int(x * scale_x)
                                        y = int(y * scale_y)
                                        x2 = int(x2 * scale_x)
                                        y2 = int(y2 * scale_y)
                                        faces.append((x, y, x2-x, y2-y))
                                else:
                                    # Convert to format [x, y, w, h]
                                    for box in boxes:
                                        x, y, x2, y2 = [int(b) for b in box]
                                        faces.append((x, y, x2-x, y2-y))
                        except Exception as e:
                            print(f"MTCNN error: {str(e)}")
                            # Fallback to OpenCV face detection if MTCNN fails
                            gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
                            cv_faces = self.face_detector.detectMultiScale(
                                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                            )
                            
                            # Rescale if necessary
                            if process_frame is not frame:
                                scale_x = w / process_frame.shape[1]
                                scale_y = h / process_frame.shape[0]
                                for (x, y, w, h) in cv_faces:
                                    faces.append((int(x*scale_x), int(y*scale_y), 
                                                int(w*scale_x), int(h*scale_y)))
                            else:
                                faces = [(x, y, w, h) for (x, y, w, h) in cv_faces]
                    else:
                        # OpenCV's detector
                        gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
                        cv_faces = self.face_detector.detectMultiScale(
                            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                        )
                        
                        # Rescale if necessary
                        if process_frame is not frame:
                            scale_x = w / process_frame.shape[1]
                            scale_y = h / process_frame.shape[0]
                            for (x, y, w, h) in cv_faces:
                                faces.append((int(x*scale_x), int(y*scale_y), 
                                            int(w*scale_x), int(h*scale_y)))
                        else:
                            faces = [(x, y, w, h) for (x, y, w, h) in cv_faces]
                
                # Process detected faces
                for (x, y, w, h) in faces:
                    # Draw rectangle
                    color = (0, 255, 0) if self.is_collecting_real else (
                        (0, 0, 255) if self.is_collecting_fake else (128, 128, 128)
                    )
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Check blur and save if collecting
                    if self.is_collecting_real or self.is_collecting_fake:
                        # Extract face with padding
                        padding = int(w * 0.2)
                        x1 = max(0, x - padding)
                        y1 = max(0, y - padding)
                        x2 = min(frame.shape[1], x + w + padding)
                        y2 = min(frame.shape[0], y + h + padding)
                        face_img = frame[y1:y2, x1:x2]
                        
                        # Check blur
                        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                        blur = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                        
                        if blur >= self.blur_threshold.get():
                            # Save face every 15 frames
                            if self.processed_count % 15 == 0:
                                self.save_face(face_img)
                        
                        # Show blur value
                        cv2.putText(frame, f"Blur: {int(blur)}", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (0, 255, 0) if blur >= self.blur_threshold.get() else (0, 0, 255),
                                  2)
                
                # Add collection status
                if self.is_collecting_real:
                    cv2.putText(frame, "Collecting: REAL", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif self.is_collecting_fake:
                    cv2.putText(frame, "Collecting: FAKE", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add FPS display
                if self.camera_thread:
                    fps_text = f"Camera: {self.camera_thread.fps:.1f} FPS | Processing: {fps:.1f} FPS"
                else:
                    fps_text = f"Processing: {fps:.1f} FPS"
                cv2.putText(frame, fps_text, (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Convert to RGB for tkinter
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Add to display queue
                if not self.frame_queue.full():
                    self.frame_queue.put(rgb_frame)
                
                # Update counters periodically
                if self.processed_count % 30 == 0:
                    self.update_counters()
                    
            except Exception as e:
                print(f"Frame processing error: {str(e)}")
                time.sleep(0.1)  # Don't flood console with errors

    def update_display(self):
        """Update the video display"""
        while not self.stop_event.is_set():
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    # Get canvas dimensions
                    canvas_width = self.canvas.winfo_width()
                    canvas_height = self.canvas.winfo_height()
                    
                    if canvas_width > 1 and canvas_height > 1:
                        # Resize frame to fit canvas
                        aspect_ratio = frame.shape[1] / frame.shape[0]
                        
                        if canvas_width / canvas_height > aspect_ratio:
                            new_height = canvas_height
                            new_width = int(new_height * aspect_ratio)
                        else:
                            new_width = canvas_width
                            new_height = int(new_width / aspect_ratio)
                        
                        frame = cv2.resize(frame, (new_width, new_height))
                        
                        # Convert to PhotoImage
                        image = Image.fromarray(frame)
                        photo = ImageTk.PhotoImage(image=image)
                        
                        # Update canvas
                        self.canvas.config(width=new_width, height=new_height)
                        self.canvas.create_image(new_width//2, new_height//2, 
                                              image=photo, anchor=tk.CENTER)
                        self.canvas.image = photo
            
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Display error: {str(e)}")
            
            time.sleep(0.03)  # ~30 FPS
    
    def save_face(self, face_img):
        """Save face image with appropriate label"""
        try:
            # Determine label and update counter
            if self.is_collecting_real:
                label = "real"
                self.real_count += 1
            elif self.is_collecting_fake:
                label = "fake"
                self.fake_count += 1
            else:
                return
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}.jpg"
            
            # Save image
            save_dir = os.path.join(self.save_location.get(), label)
            os.makedirs(save_dir, exist_ok=True)
            
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, face_img)
            
            # Save metadata
            meta_filepath = os.path.join(save_dir, f"{timestamp}.json")
            metadata = {
                "label": label,
                "timestamp": timestamp,
                "resolution": f"{face_img.shape[1]}x{face_img.shape[0]}",
                "source": f"Camera {self.camera_source.get()}"
            }
            
            with open(meta_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update display
            self.update_counters()
            
        except Exception as e:
            print(f"Error saving image: {str(e)}")
    
    def update_counters(self):
        """Update the sample counters"""
        self.real_label.config(text=f"Real: {self.real_count}")
        self.fake_label.config(text=f"Fake: {self.fake_count}")
    
    def on_close(self):
        """Handle window close"""
        if self.is_recording:
            self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FaceDataCollectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main() 