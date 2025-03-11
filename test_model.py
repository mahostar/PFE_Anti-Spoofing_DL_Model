"""
DepthFusion-ViT Model Testing Tool
---------------------------------
This script provides a real-time testing interface for the trained
anti-spoofing model. It shows face detection, anti-spoofing scores,
and prediction confidence in real-time.
"""

import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
import time
import os


class ModelTester:
    def __init__(self, root):
        self.root = root
        self.root.title("DepthFusion-ViT Model Tester")
        self.root.geometry("1200x800")
        
        # Variables
        self.model = None
        self.model_path = None
        self.camera_source = tk.IntVar(value=0)
        self.threshold = tk.DoubleVar(value=0.5)
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        
        # Initialize face detector
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Create GUI
        self.create_widgets()
        
        # Load model if available
        self.try_load_default_model()
    
    def create_widgets(self):
        # Main layout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Model selection
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT)
        self.model_label = ttk.Label(model_frame, text="No model loaded")
        self.model_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="Load Model", 
                  command=self.load_model).pack(side=tk.RIGHT)
        
        # Camera selection
        camera_frame = ttk.Frame(control_frame)
        camera_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(camera_frame, text="Camera:").pack(side=tk.LEFT)
        ttk.Combobox(camera_frame, textvariable=self.camera_source,
                    values=[0, 1, 2, 3]).pack(side=tk.LEFT, padx=5)
        
        # Threshold control
        threshold_frame = ttk.Frame(control_frame)
        threshold_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(threshold_frame, text="Threshold:").pack(side=tk.LEFT)
        ttk.Scale(threshold_frame, from_=0.0, to=1.0, 
                 variable=self.threshold, orient=tk.HORIZONTAL).pack(
                     side=tk.LEFT, fill=tk.X, expand=True)
        
        # Start/Stop button
        self.start_button = ttk.Button(control_frame, text="Start Testing",
                                     command=self.toggle_testing)
        self.start_button.pack(fill=tk.X, padx=5, pady=5)
        
        # Statistics
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=8, width=30)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel - Video display
        video_frame = ttk.LabelFrame(main_frame, text="Camera Preview")
        video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(video_frame)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def try_load_default_model(self):
        """Try to load model from default location"""
        default_path = os.path.join(os.path.dirname(__file__), "models", 
                                  "depthfusion_vit.h5")
        if os.path.exists(default_path):
            self.load_model_from_path(default_path)
    
    def load_model(self):
        """Open file dialog to select model file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Model Files", "*.h5"), ("All Files", "*.*")]
        )
        if file_path:
            self.load_model_from_path(file_path)
    
    def load_model_from_path(self, path):
        """Load the TensorFlow model from file"""
        try:
            self.model = tf.keras.models.load_model(path)
            self.model_path = path
            self.model_label.config(
                text=os.path.basename(path)
            )
            self.status_var.set("Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def toggle_testing(self):
        """Start or stop testing"""
        if not self.is_running:
            if self.model is None:
                messagebox.showinfo("Info", "Please load a model first")
                return
            
            self.start_testing()
            self.start_button.config(text="Stop Testing")
        else:
            self.stop_testing()
            self.start_button.config(text="Start Testing")
    
    def start_testing(self):
        """Start the camera and testing process"""
        try:
            # Open camera
            self.cap = cv2.VideoCapture(self.camera_source.get())
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Start processing
            self.is_running = True
            self.stop_event.clear()
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.process_camera)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            # Start display thread
            self.display_thread = threading.Thread(target=self.update_display)
            self.display_thread.daemon = True
            self.display_thread.start()
            
            self.status_var.set("Testing started")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.is_running = False
    
    def stop_testing(self):
        """Stop testing"""
        self.is_running = False
        self.stop_event.set()
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        self.canvas.delete("all")
        self.status_var.set("Testing stopped")
    
    def preprocess_face(self, face_img):
        """Preprocess face image for model input"""
        # Resize to model input size
        face_resized = cv2.resize(face_img, (224, 224))
        
        # Convert to float and normalize
        face_float = face_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        face_batch = np.expand_dims(face_float, axis=0)
        
        return face_batch
    
    def process_camera(self):
        """Process camera frames and run inference"""
        fps_time = time.time()
        fps = 0
        frames = 0
        
        while not self.stop_event.is_set() and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Process each face
            for (x, y, w, h) in faces:
                # Extract face with padding
                padding = int(w * 0.2)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                face_img = frame[y1:y2, x1:x2]
                
                # Preprocess and run inference
                face_input = self.preprocess_face(face_img)
                prediction = self.model.predict(face_input, verbose=0)[0][0]
                
                # Draw results
                color = (0, 255, 0) if prediction >= self.threshold.get() else (0, 0, 255)
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add prediction text
                text = f"Real: {prediction:.2%}"
                cv2.putText(rgb_frame, text, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Calculate FPS
            frames += 1
            if time.time() - fps_time > 1.0:
                fps = frames
                frames = 0
                fps_time = time.time()
            
            # Add FPS to frame
            cv2.putText(rgb_frame, f"FPS: {fps}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Update display queue
            if not self.frame_queue.full():
                self.frame_queue.put(rgb_frame)
    
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
    
    def on_close(self):
        """Handle window close"""
        if self.is_running:
            self.stop_testing()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ModelTester(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main() 