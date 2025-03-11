"""
Test script to verify GPU setup and model architecture
"""

import os
import tensorflow as tf
import torch
import cv2
import numpy as np
from datetime import datetime

def test_tensorflow_gpu():
    """Test TensorFlow GPU availability and performance"""
    print("\n=== TensorFlow GPU Test ===")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    print("\nGPU Devices:", gpus)
    
    if not gpus:
        print("No GPU devices found for TensorFlow!")
        return False
    
    # Test GPU performance
    print("\nTesting GPU Performance...")
    
    # Create and process a large tensor
    with tf.device('/GPU:0'):
        start_time = datetime.now()
        
        # Matrix multiplication test
        matrix_size = 2000
        a = tf.random.normal([matrix_size, matrix_size])
        b = tf.random.normal([matrix_size, matrix_size])
        c = tf.matmul(a, b)
        
        # Force execution
        result = c.numpy()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"Matrix multiplication ({matrix_size}x{matrix_size}) took: {duration:.2f} seconds")
    
    return True

def test_pytorch_gpu():
    """Test PyTorch GPU availability and performance"""
    print("\n=== PyTorch GPU Test ===")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print("\nCUDA Available:", cuda_available)
    
    if not cuda_available:
        print("CUDA not available for PyTorch!")
        return False
    
    # Print CUDA device info
    device = torch.device("cuda")
    print("CUDA Device:", torch.cuda.get_device_name(0))
    
    # Test GPU performance
    print("\nTesting GPU Performance...")
    
    # Create and process a large tensor
    start_time = datetime.now()
    
    # Matrix multiplication test
    matrix_size = 2000
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)
    c = torch.matmul(a, b)
    
    # Force execution
    result = c.cpu().numpy()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"Matrix multiplication ({matrix_size}x{matrix_size}) took: {duration:.2f} seconds")
    
    return True

def test_opencv():
    """Test OpenCV installation"""
    print("\n=== OpenCV Test ===")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Try to open camera
    print("\nTesting camera access...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Could not open camera!")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Warning: Could not read frame from camera!")
        return False
    
    print("Successfully accessed camera and captured frame")
    print(f"Frame shape: {frame.shape}")
    return True

def main():
    """Run all tests"""
    print("=== Starting Environment Tests ===")
    print(f"Python version: {sys.version}")
    
    # Test TensorFlow
    tf_success = test_tensorflow_gpu()
    
    # Test PyTorch
    torch_success = test_pytorch_gpu()
    
    # Test OpenCV
    opencv_success = test_opencv()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"TensorFlow GPU: {'✓' if tf_success else '✗'}")
    print(f"PyTorch GPU: {'✓' if torch_success else '✗'}")
    print(f"OpenCV: {'✓' if opencv_success else '✗'}")

if __name__ == "__main__":
    import sys
    main() 