#!/usr/bin/env python3
"""
Improved Face Recognition System
Handles multiple images of the same person and better recognition accuracy.
"""

import os
import cv2
import numpy as np
import pickle
from pathlib import Path
import time

class ImprovedFaceRecognizer:
    def __init__(self, known_faces_dir="known_faces", model_file="improved_face_model.pkl"):
        """Initialize the improved face recognition system."""
        self.known_faces_dir = Path(known_faces_dir)
        self.model_file = model_file
        
        # Initialize OpenCV components
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        self.person_names = []
        self.is_trained = False
        
        # Recognition settings - more lenient for better accuracy
        self.confidence_threshold = 120  # Increased for better recognition
        
        # Load or train model
        if not self.load_model():
            self.train_recognizer()
    
    def extract_person_name(self, filename):
        """Extract person name from filename, handling multiple images of same person."""
        # Remove extension and numbers/suffixes
        name = Path(filename).stem.lower()
        
        # Remove common suffixes like _2, _test, etc.
        import re
        name = re.sub(r'[_\-]\d+$', '', name)  # Remove _2, _3, etc.
        name = re.sub(r'[_\-]test[_\-]?\d*$', '', name)  # Remove _test, _test_2, etc.
        name = re.sub(r'[_\-]photo[_\-]?\d*$', '', name)  # Remove _photo, etc.
        
        return name.capitalize()
    
    def load_and_group_faces(self):
        """Load faces and group multiple images of the same person."""
        print(f"Loading faces from {self.known_faces_dir}...")
        
        if not self.known_faces_dir.exists():
            self.known_faces_dir.mkdir(exist_ok=True)
            return [], [], []
        
        person_faces = {}  # Dictionary to group faces by person
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Group images by person name
        for image_path in self.known_faces_dir.iterdir():
            if image_path.suffix.lower() in supported_formats:
                person_name = self.extract_person_name(image_path.name)
                
                if person_name not in person_faces:
                    person_faces[person_name] = []
                
                person_faces[person_name].append(image_path)
        
        # Process each person's images
        all_faces = []
        all_labels = []
        person_names = []
        
        for person_id, (person_name, image_paths) in enumerate(person_faces.items()):
            person_names.append(person_name)
            print(f"Processing {person_name} ({len(image_paths)} images):")
            
            for image_path in image_paths:
                try:
                    # Load and process image
                    image = cv2.imread(str(image_path))
                    if image is None:
                        print(f"  ✗ Could not load: {image_path.name}")
                        continue
                    
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )
                    
                    if len(faces) > 0:
                        # Use the largest face
                        largest_face = max(faces, key=lambda face: face[2] * face[3])
                        x, y, w, h = largest_face
                        
                        # Extract and resize face
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (100, 100))
                        
                        all_faces.append(face_roi)
                        all_labels.append(person_id)
                        
                        print(f"  ✓ Loaded: {image_path.name}")
                    else:
                        print(f"  ✗ No face found in: {image_path.name}")
                        
                except Exception as e:
                    print(f"  ✗ Error loading {image_path.name}: {str(e)}")
        
        print(f"Total: {len(all_faces)} face samples from {len(person_names)} people")
        return all_faces, all_labels, person_names
    
    def train_recognizer(self):
        """Train the face recognizer."""
        faces, labels, person_names = self.load_and_group_faces()
        
        if len(faces) == 0:
            print("No faces to train with.")
            return False
        
        print("Training face recognizer...")
        self.face_recognizer.train(faces, np.array(labels))
        self.person_names = person_names
        self.is_trained = True
        
        # Save model
        self.save_model()
        print("Training completed!")
        return True
    
    def save_model(self):
        """Save the trained model."""
        try:
            self.face_recognizer.save("improved_face_model.yml")
            
            with open(self.model_file, 'wb') as f:
                pickle.dump({
                    'person_names': self.person_names,
                    'is_trained': self.is_trained
                }, f)
            
            print(f"Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load existing model."""
        try:
            if os.path.exists("improved_face_model.yml") and os.path.exists(self.model_file):
                self.face_recognizer.read("improved_face_model.yml")
                
                with open(self.model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.person_names = data['person_names']
                    self.is_trained = data['is_trained']
                
                print(f"✓ Loaded model with {len(self.person_names)} people:")
                for name in self.person_names:
                    print(f"  - {name}")
                return True
        except Exception as e:
            print(f"Could not load model: {str(e)}")
        
        return False
    
    def recognize_face(self, face_roi):
        """Recognize a single face region."""
        if not self.is_trained:
            return "No Model", 0.0
        
        try:
            # Resize face to standard size
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Predict
            label, confidence = self.face_recognizer.predict(face_roi)
            
            # Convert confidence (lower is better) to similarity
            if confidence < self.confidence_threshold:
                name = self.person_names[label] if label < len(self.person_names) else "Unknown"
                similarity = max(0, (self.confidence_threshold - confidence) / self.confidence_threshold)
            else:
                name = "Unknown"
                similarity = 0.0
            
            return name, similarity
            
        except Exception as e:
            return "Error", 0.0
    
    def detect_and_recognize_faces(self, frame):
        """Detect and recognize faces in a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        
        results = []
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Recognize face
            name, confidence = self.recognize_face(face_roi)
            
            # Determine color based on recognition
            if name == "Unknown" or name == "Error" or name == "No Model":
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)  # Green
            
            results.append({
                'name': name,
                'confidence': confidence,
                'location': (x, y, w, h),
                'color': color
            })
        
        return results
    
    def draw_results(self, frame, results):
        """Draw recognition results on frame."""
        for result in results:
            x, y, w, h = result['location']
            name = result['name']
            confidence = result['confidence']
            color = result['color']
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label
            if name in ["Unknown", "Error", "No Model"]:
                label = name
            else:
                label = f"{name} ({confidence:.2f})"
            
            # Draw label background and text
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2
            )
            
            cv2.rectangle(
                frame, (x, y - text_height - 10), (x + text_width, y), color, cv2.FILLED
            )
            cv2.putText(
                frame, label, (x, y - 5), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2
            )
    
    def run_camera(self):
        """Run camera recognition."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open camera")
            return
        
        print("\n" + "="*50)
        print("IMPROVED FACE RECOGNITION CAMERA")
        print("="*50)
        print("Controls: 'q' to quit, 's' to screenshot, 'r' to retrain")
        print("="*50)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect and recognize
                results = self.detect_and_recognize_faces(frame)
                
                # Draw results
                self.draw_results(frame, results)
                
                # Show status
                status = f"Ready - {len(self.person_names)} people" if self.is_trained else "No model trained"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Improved Face Recognition', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"screenshot_{timestamp}.jpg", frame)
                    print(f"Screenshot saved: screenshot_{timestamp}.jpg")
                elif key == ord('r'):
                    print("Retraining...")
                    self.train_recognizer()
                    
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = ImprovedFaceRecognizer()
    recognizer.run_camera()
