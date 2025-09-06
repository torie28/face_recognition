#!/usr/bin/env python3
"""
Real-time Camera Face Recognition System
Opens camera to detect and recognize faces in real-time using trained models.
"""

import os
import cv2
import numpy as np
import pickle
from pathlib import Path
import time

class CameraFaceRecognizer:
    def __init__(self, known_faces_dir="known_faces", model_file="face_model.pkl"):
        """
        Initialize the camera face recognition system.
        
        Args:
            known_faces_dir (str): Directory containing known face images
            model_file (str): File to load the trained model
        """
        self.known_faces_dir = Path(known_faces_dir)
        self.model_file = model_file
        
        # Initialize OpenCV face detector and recognizer
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        self.label_to_name = {}
        self.is_trained = False
        
        # Load existing model
        self.load_model()
        
        # Camera settings
        self.camera = None
        self.camera_width = 640
        self.camera_height = 480
        
        # Recognition settings
        self.confidence_threshold = 100  # Lower = more strict, increased for better recognition
        self.recognition_delay = 0.1  # Delay between recognitions (seconds)
        self.last_recognition_time = 0
        
        # Display settings
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale = 0.8
        self.thickness = 2
        
    def load_model(self):
        """Load a previously trained model."""
        try:
            if os.path.exists("opencv_face_model.yml") and os.path.exists(self.model_file):
                self.face_recognizer.read("opencv_face_model.yml")
                
                with open(self.model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.label_to_name = data['label_to_name']
                    self.is_trained = data['is_trained']
                
                print(f"✓ Loaded model with {len(self.label_to_name)} known faces:")
                for name in self.label_to_name.values():
                    print(f"  - {name}")
                return True
        except Exception as e:
            print(f"Could not load existing model: {str(e)}")
        
        return False
    
    def initialize_camera(self, camera_index=0):
        """Initialize the camera."""
        try:
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                raise Exception(f"Could not open camera {camera_index}")
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"✓ Camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize camera: {str(e)}")
            return False
    
    def detect_and_recognize_faces(self, frame):
        """
        Detect and recognize faces in a frame.
        
        Args:
            frame: Camera frame (BGR image)
            
        Returns:
            list: List of dictionaries containing face information
        """
        if not self.is_trained:
            return []
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        results = []
        current_time = time.time()
        
        # Skip recognition if too soon since last recognition (for performance)
        if current_time - self.last_recognition_time < self.recognition_delay:
            # Return previous results with updated positions
            for (x, y, w, h) in faces:
                results.append({
                    'name': "Processing...",
                    'confidence': 0.0,
                    'location': (x, y, w, h),
                    'color': (255, 255, 0)  # Yellow for processing
                })
            return results
        
        self.last_recognition_time = current_time
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to standard size for better recognition
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Predict
            try:
                label, confidence = self.face_recognizer.predict(face_roi)
                
                # Determine name and color based on confidence
                if confidence < self.confidence_threshold:
                    name = self.label_to_name.get(label, "Unknown")
                    similarity = max(0, (100 - confidence) / 100)
                    color = (0, 255, 0)  # Green for recognized
                else:
                    name = "Unknown"
                    similarity = 0.0
                    color = (0, 0, 255)  # Red for unknown
                
                results.append({
                    'name': name,
                    'confidence': similarity,
                    'location': (x, y, w, h),
                    'color': color,
                    'raw_confidence': confidence
                })
                
            except Exception as e:
                # If recognition fails, mark as unknown
                results.append({
                    'name': "Error",
                    'confidence': 0.0,
                    'location': (x, y, w, h),
                    'color': (0, 0, 255)
                })
        
        return results
    
    def draw_face_info(self, frame, results):
        """
        Draw face detection and recognition results on the frame.
        
        Args:
            frame: Camera frame to draw on
            results: List of face detection results
        """
        for result in results:
            x, y, w, h = result['location']
            name = result['name']
            confidence = result['confidence']
            color = result['color']
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, self.thickness)
            
            # Prepare label text
            if name == "Processing...":
                label = name
            elif name == "Unknown" or name == "Error":
                label = name
            else:
                label = f"{name} ({confidence:.2f})"
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.thickness
            )
            
            # Draw label background
            cv2.rectangle(
                frame,
                (x, y - text_height - 10),
                (x + text_width, y),
                color,
                cv2.FILLED
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x, y - 5),
                self.font,
                self.font_scale,
                (255, 255, 255),
                self.thickness
            )
            
            # Add confidence bar for recognized faces
            if name != "Unknown" and name != "Error" and name != "Processing...":
                bar_width = int(w * confidence)
                cv2.rectangle(
                    frame,
                    (x, y + h + 5),
                    (x + bar_width, y + h + 15),
                    color,
                    cv2.FILLED
                )
    
    def draw_status_info(self, frame):
        """Draw status information on the frame."""
        height, width = frame.shape[:2]
        
        # Status text
        if not self.is_trained:
            status = "No trained model - Add faces to known_faces/ and retrain"
            color = (0, 0, 255)
        else:
            status = f"Ready - {len(self.label_to_name)} known faces loaded"
            color = (0, 255, 0)
        
        # Draw status background
        (text_width, text_height), _ = cv2.getTextSize(
            status, self.font, 0.6, 1
        )
        cv2.rectangle(
            frame,
            (10, 10),
            (text_width + 20, text_height + 20),
            (0, 0, 0),
            cv2.FILLED
        )
        
        # Draw status text
        cv2.putText(
            frame,
            status,
            (15, text_height + 15),
            self.font,
            0.6,
            color,
            1
        )
        
        # Instructions
        instructions = [
            "Press 'q' to quit",
            "Press 's' to save screenshot",
            "Press 'r' to reload model"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = height - (len(instructions) - i) * 25
            cv2.putText(
                frame,
                instruction,
                (10, y_pos),
                self.font,
                0.5,
                (255, 255, 255),
                1
            )
    
    def save_screenshot(self, frame):
        """Save a screenshot of the current frame."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"camera_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
    
    def run_camera_recognition(self):
        """Main loop for camera-based face recognition."""
        if not self.initialize_camera():
            return False
        
        print("\n" + "="*50)
        print("CAMERA FACE RECOGNITION STARTED")
        print("="*50)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'r' - Reload model")
        print("="*50)
        
        try:
            while True:
                # Read frame from camera
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect and recognize faces
                results = self.detect_and_recognize_faces(frame)
                
                # Draw face information
                self.draw_face_info(frame, results)
                
                # Draw status information
                self.draw_status_info(frame)
                
                # Display the frame
                cv2.imshow('Face Recognition Camera', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    self.save_screenshot(frame)
                elif key == ord('r'):
                    print("Reloading model...")
                    self.load_model()
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed")
        
        return True

def main():
    """Main function to run the camera face recognition system."""
    print("Initializing Camera Face Recognition System...")
    
    # Create recognizer instance
    recognizer = CameraFaceRecognizer()
    
    # Check if model is trained
    if not recognizer.is_trained:
        print("\n⚠️  WARNING: No trained model found!")
        print("Please ensure you have:")
        print("1. Face images in the 'known_faces/' directory")
        print("2. Run training with: python opencv_face_recognition.py retrain")
        print("\nContinuing anyway - camera will show 'No trained model' status")
    
    # Run camera recognition
    success = recognizer.run_camera_recognition()
    
    if not success:
        print("Failed to start camera recognition system")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
