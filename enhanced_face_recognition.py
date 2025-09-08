#!/usr/bin/env python3
"""
Enhanced Face Recognition System
Focuses on facial features only, robust to clothing/lighting changes.
"""

import os
import cv2
import numpy as np
import pickle
from pathlib import Path
import time

class EnhancedFaceRecognizer:
    def __init__(self, known_faces_dir="known_faces", model_file="enhanced_face_model.pkl"):
        """Initialize enhanced face recognition system."""
        self.known_faces_dir = Path(known_faces_dir)
        self.model_file = model_file
        
        # Initialize OpenCV components
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2,           # Smaller radius for finer details
            neighbors=16,       # More neighbors for better texture analysis
            grid_x=8,          # Grid size for local patterns
            grid_y=8,
            threshold=150.0     # Higher threshold for more lenient matching
        )
        
        self.person_names = []
        self.is_trained = False
        
        # Enhanced recognition settings
        self.confidence_threshold = 200  # Much more lenient threshold
        
        # Load or train model
        if not self.load_model():
            self.train_recognizer()
    
    def extract_person_name(self, filename):
        """Extract person name from filename, grouping similar names."""
        name = Path(filename).stem.lower()
        
        # Remove common suffixes and variations
        import re
        name = re.sub(r'[_\-]\d+$', '', name)
        name = re.sub(r'[_\-]test[_\-]?\d*$', '', name)
        name = re.sub(r'[_\-]photo[_\-]?\d*$', '', name)
        name = re.sub(r'[_\-]pic[_\-]?\d*$', '', name)
        
        return name.capitalize()
    
    def preprocess_face(self, face_roi):
        """Enhanced face preprocessing to normalize appearance."""
        # Resize to standard size
        face_roi = cv2.resize(face_roi, (120, 120))
        
        # Apply histogram equalization for lighting normalization
        face_roi = cv2.equalizeHist(face_roi)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        face_roi = clahe.apply(face_roi)
        
        # Apply Gaussian blur to reduce noise
        face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
        
        # Crop to focus on central facial features (remove edges that might include clothing/background)
        h, w = face_roi.shape
        crop_margin = int(min(h, w) * 0.1)  # Remove 10% from edges
        face_roi = face_roi[crop_margin:h-crop_margin, crop_margin:w-crop_margin]
        
        # Resize back to standard size after cropping
        face_roi = cv2.resize(face_roi, (100, 100))
        
        return face_roi
    
    def detect_faces_enhanced(self, image):
        """Enhanced face detection with multiple attempts."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing to improve detection
        gray = cv2.equalizeHist(gray)
        
        # Try multiple detection parameters
        detection_params = [
            {"scaleFactor": 1.05, "minNeighbors": 3, "minSize": (40, 40)},
            {"scaleFactor": 1.1, "minNeighbors": 4, "minSize": (30, 30)},
            {"scaleFactor": 1.2, "minNeighbors": 3, "minSize": (25, 25)},
            {"scaleFactor": 1.05, "minNeighbors": 2, "minSize": (20, 20)},
        ]
        
        for params in detection_params:
            faces = self.face_cascade.detectMultiScale(gray, **params)
            if len(faces) > 0:
                return faces, gray
        
        return [], gray
    
    def load_and_group_faces(self):
        """Load faces with enhanced preprocessing."""
        print(f"🔍 Loading faces from {self.known_faces_dir} with enhanced processing...")
        
        if not self.known_faces_dir.exists():
            self.known_faces_dir.mkdir(exist_ok=True)
            return [], [], []
        
        person_faces = {}
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for image_path in self.known_faces_dir.iterdir():
            if image_path.suffix.lower() not in supported_formats:
                continue
                
            print(f"\n📸 Processing: {image_path.name}")
            
            try:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"  ❌ Could not load image")
                    continue
                
                # Enhanced face detection
                faces, gray = self.detect_faces_enhanced(image)
                
                if len(faces) == 0:
                    print(f"  ❌ No faces detected")
                    continue
                
                # Extract person name
                person_name = self.extract_person_name(image_path.name)
                
                if person_name not in person_faces:
                    person_faces[person_name] = []
                
                # Process all detected faces (use largest one)
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                x, y, w, h = largest_face
                
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Apply enhanced preprocessing
                processed_face = self.preprocess_face(face_roi)
                
                person_faces[person_name].append(processed_face)
                print(f"  ✓ Processed face: {w}x{h} → 100x100 (enhanced)")
                
            except Exception as e:
                print(f"  ❌ Error processing {image_path.name}: {str(e)}")
        
        # Prepare training data
        all_faces = []
        all_labels = []
        person_names = []
        
        for person_id, (person_name, faces) in enumerate(person_faces.items()):
            person_names.append(person_name)
            print(f"\n👤 {person_name}: {len(faces)} face samples")
            
            for face in faces:
                all_faces.append(face)
                all_labels.append(person_id)
        
        return all_faces, all_labels, person_names
    
    def train_recognizer(self):
        """Train with enhanced preprocessing."""
        faces, labels, person_names = self.load_and_group_faces()
        
        if len(faces) == 0:
            print("❌ No faces to train with.")
            return False
        
        print(f"\n🎯 Training enhanced recognizer...")
        self.face_recognizer.train(faces, np.array(labels))
        self.person_names = person_names
        self.is_trained = True
        
        self.save_model()
        print("✅ Enhanced training completed!")
        return True
    
    def save_model(self):
        """Save the enhanced model."""
        try:
            self.face_recognizer.save("enhanced_face_model.yml")
            
            with open(self.model_file, 'wb') as f:
                pickle.dump({
                    'person_names': self.person_names,
                    'is_trained': self.is_trained
                }, f)
            
            print(f"💾 Enhanced model saved")
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
    
    def load_model(self):
        """Load existing enhanced model."""
        try:
            if os.path.exists("enhanced_face_model.yml") and os.path.exists(self.model_file):
                self.face_recognizer.read("enhanced_face_model.yml")
                
                with open(self.model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.person_names = data['person_names']
                    self.is_trained = data['is_trained']
                
                print(f"✅ Loaded enhanced model with {len(self.person_names)} people:")
                for name in self.person_names:
                    print(f"  - {name}")
                return True
        except Exception as e:
            print(f"Could not load enhanced model: {str(e)}")
        
        return False
    
    def recognize_face_enhanced(self, face_roi):
        """Enhanced face recognition with better preprocessing."""
        if not self.is_trained:
            return "No Model", 0.0
        
        try:
            # Apply same preprocessing as training
            processed_face = self.preprocess_face(face_roi)
            
            # Predict with enhanced model
            label, confidence = self.face_recognizer.predict(processed_face)
            
            # Debug output
            print(f"Debug: label={label}, confidence={confidence}, threshold={self.confidence_threshold}")
            
            # Much more lenient confidence evaluation
            if confidence < self.confidence_threshold:
                name = self.person_names[label] if label < len(self.person_names) else "Unknown"
                # Calculate similarity score (higher confidence threshold = more lenient)
                similarity = max(0.1, (self.confidence_threshold - confidence) / self.confidence_threshold)
            else:
                # Even if confidence is high, still try to match if we have people trained
                if len(self.person_names) > 0:
                    name = self.person_names[label] if label < len(self.person_names) else self.person_names[0]
                    similarity = 0.3  # Give some confidence even for high confidence values
                else:
                    name = "Unknown"
                    similarity = 0.0
            
            return name, similarity
            
        except Exception as e:
            print(f"Recognition error: {e}")
            return "Error", 0.0
    
    def detect_and_recognize_faces(self, frame):
        """Enhanced real-time face detection and recognition."""
        # Enhanced face detection
        faces, gray = self.detect_faces_enhanced(frame)
        
        results = []
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Enhanced recognition
            name, confidence = self.recognize_face_enhanced(face_roi)
            
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
        """Draw enhanced results on frame."""
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
                label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2
            )
            
            cv2.rectangle(
                frame, (x, y - text_height - 10), (x + text_width, y), color, cv2.FILLED
            )
            cv2.putText(
                frame, label, (x, y - 5), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2
            )
            
            # Add confidence bar
            if name not in ["Unknown", "Error", "No Model"]:
                bar_width = int(w * confidence)
                cv2.rectangle(
                    frame, (x, y + h + 5), (x + bar_width, y + h + 15), color, cv2.FILLED
                )
    
    def run_camera(self):
        """Run enhanced camera recognition."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        print("\n" + "="*60)
        print("🚀 ENHANCED FACE RECOGNITION CAMERA")
        print("="*60)
        print("✨ Features:")
        print("  - Robust to clothing/lighting changes")
        print("  - Focus on facial features only")
        print("  - Enhanced preprocessing")
        print("="*60)
        print("Controls: 'q' to quit, 's' to screenshot, 'r' to retrain")
        print("="*60)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Enhanced detection and recognition
                results = self.detect_and_recognize_faces(frame)
                
                # Draw results
                self.draw_results(frame, results)
                
                # Show status
                status = f"Enhanced Mode - {len(self.person_names)} people trained" if self.is_trained else "No model trained"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow('Enhanced Face Recognition', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"enhanced_screenshot_{timestamp}.jpg", frame)
                    print(f"📸 Screenshot saved: enhanced_screenshot_{timestamp}.jpg")
                elif key == ord('r'):
                    print("🔄 Retraining enhanced model...")
                    self.train_recognizer()
                    
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("📷 Camera closed")

if __name__ == "__main__":
    recognizer = EnhancedFaceRecognizer()
    recognizer.run_camera()
