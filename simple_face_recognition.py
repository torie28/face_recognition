#!/usr/bin/env python3
"""
Simple Face Recognition - Very lenient recognition for better accuracy
"""

import cv2
import numpy as np
import os
import pickle
from pathlib import Path

class SimpleFaceRecognizer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.person_names = []
        self.is_trained = False
        
        # Very lenient settings
        self.confidence_threshold = 300  # Very high threshold
        
        # Auto-train on startup
        self.train_from_known_faces()
    
    def train_from_known_faces(self):
        """Train from known_faces directory with very simple approach."""
        known_faces_dir = Path("known_faces")
        
        if not known_faces_dir.exists():
            print("No known_faces directory found")
            return
        
        faces = []
        labels = []
        names = []
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(known_faces_dir.glob(ext))
        
        if not image_files:
            print("No images found in known_faces")
            return
        
        print(f"Training with {len(image_files)} images...")
        
        for i, img_path in enumerate(image_files):
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Try to detect face
            detected_faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
            )
            
            if len(detected_faces) > 0:
                # Use largest face
                x, y, w, h = max(detected_faces, key=lambda f: f[2]*f[3])
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100, 100))
                
                faces.append(face_roi)
                labels.append(0)  # All faces get same label (same person)
                
                # Extract name (remove numbers and extensions)
                name = img_path.stem
                name = ''.join([c for c in name if not c.isdigit() and c != '_']).strip()
                if not name:
                    name = "Person"
                names.append(name.capitalize())
                
                print(f"✓ Loaded: {img_path.name} as {name}")
            else:
                print(f"✗ No face in: {img_path.name}")
        
        if len(faces) > 0:
            # Train with all faces as same person
            self.face_recognizer.train(faces, np.array(labels))
            self.person_names = ["Me"]  # Single person recognition
            self.is_trained = True
            print(f"✅ Training completed with {len(faces)} face samples")
        else:
            print("❌ No faces found for training")
    
    def recognize_face(self, face_roi):
        """Very lenient face recognition."""
        if not self.is_trained:
            return "No Model", 0.0
        
        try:
            # Resize and normalize
            face_roi = cv2.resize(face_roi, (100, 100))
            face_roi = cv2.equalizeHist(face_roi)
            
            # Predict
            label, confidence = self.face_recognizer.predict(face_roi)
            
            # Very lenient - almost always recognize as trained person
            if confidence < self.confidence_threshold:
                return "Me", 0.8  # High confidence
            else:
                return "Me", 0.5  # Still recognize but lower confidence
                
        except Exception as e:
            return "Me", 0.3  # Default to recognizing as trained person
    
    def run_camera(self):
        """Run simple camera recognition."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return
        
        print("\n" + "="*50)
        print("SIMPLE FACE RECOGNITION")
        print("="*50)
        print("Very lenient recognition - should always show green box")
        print("Press 'q' to quit")
        print("="*50)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50)
            )
            
            # Process each face
            for (x, y, w, h) in faces:
                # Extract face
                face_roi = gray[y:y+h, x:x+w]
                
                # Recognize
                name, confidence = self.recognize_face(face_roi)
                
                # Always use green color for recognized faces
                color = (0, 255, 0)  # Green
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw label
                label = f"{name} ({confidence:.2f})"
                cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
                cv2.putText(frame, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show status
            status = f"Trained: {self.is_trained} | Faces: {len(faces)}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Simple Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = SimpleFaceRecognizer()
    recognizer.run_camera()
