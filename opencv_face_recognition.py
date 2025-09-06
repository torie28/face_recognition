#!/usr/bin/env python3
"""
OpenCV-based Face Recognition System
A face recognition system using OpenCV's built-in face detection and recognition capabilities.
No dlib dependency required.
"""

import os
import cv2
import numpy as np
from PIL import Image
import click
from pathlib import Path
import pickle

class OpenCVFaceRecognizer:
    def __init__(self, known_faces_dir="known_faces", model_file="face_model.pkl"):
        """
        Initialize the OpenCV face recognition system.
        
        Args:
            known_faces_dir (str): Directory containing known face images
            model_file (str): File to save/load the trained model
        """
        self.known_faces_dir = Path(known_faces_dir)
        self.model_file = model_file
        
        # Initialize OpenCV face detector and recognizer
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        self.known_face_labels = []
        self.label_to_name = {}
        self.is_trained = False
        
        # Load existing model if available
        self.load_model()
        
        # If no model exists, train with known faces
        if not self.is_trained:
            self.train_recognizer()
    
    def detect_faces(self, image):
        """Detect faces in an image using OpenCV."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces, gray
    
    def load_known_faces(self):
        """Load and prepare known faces for training."""
        print(f"Loading known faces from {self.known_faces_dir}...")
        
        if not self.known_faces_dir.exists():
            print(f"Creating {self.known_faces_dir} directory...")
            self.known_faces_dir.mkdir(exist_ok=True)
            return [], []
        
        faces = []
        labels = []
        label_counter = 0
        
        # Supported image formats
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for image_path in self.known_faces_dir.iterdir():
            if image_path.suffix.lower() in supported_formats:
                try:
                    # Load image
                    image = cv2.imread(str(image_path))
                    if image is None:
                        print(f"✗ Could not load: {image_path.name}")
                        continue
                    
                    # Detect faces
                    detected_faces, gray = self.detect_faces(image)
                    
                    if len(detected_faces) > 0:
                        # Use the largest face found
                        largest_face = max(detected_faces, key=lambda face: face[2] * face[3])
                        x, y, w, h = largest_face
                        
                        # Extract face region
                        face_roi = gray[y:y+h, x:x+w]
                        
                        # Resize to standard size
                        face_roi = cv2.resize(face_roi, (100, 100))
                        
                        faces.append(face_roi)
                        labels.append(label_counter)
                        
                        name = image_path.stem
                        self.label_to_name[label_counter] = name
                        
                        print(f"✓ Loaded face: {name}")
                        label_counter += 1
                    else:
                        print(f"✗ No face found in: {image_path.name}")
                        
                except Exception as e:
                    print(f"✗ Error loading {image_path.name}: {str(e)}")
        
        print(f"Loaded {len(faces)} known faces.")
        return faces, labels
    
    def train_recognizer(self):
        """Train the face recognizer with known faces."""
        faces, labels = self.load_known_faces()
        
        if len(faces) == 0:
            print("No faces to train with. Add face images to the known_faces directory.")
            return False
        
        print("Training face recognizer...")
        self.face_recognizer.train(faces, np.array(labels))
        self.is_trained = True
        
        # Save the trained model
        self.save_model()
        print("Training completed and model saved.")
        return True
    
    def save_model(self):
        """Save the trained model and labels."""
        try:
            self.face_recognizer.save(f"opencv_face_model.yml")
            
            # Save label mappings
            with open(self.model_file, 'wb') as f:
                pickle.dump({
                    'label_to_name': self.label_to_name,
                    'is_trained': self.is_trained
                }, f)
            
            print(f"Model saved to opencv_face_model.yml and {self.model_file}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load a previously trained model."""
        try:
            if os.path.exists("opencv_face_model.yml") and os.path.exists(self.model_file):
                self.face_recognizer.read("opencv_face_model.yml")
                
                with open(self.model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.label_to_name = data['label_to_name']
                    self.is_trained = data['is_trained']
                
                print(f"Loaded existing model with {len(self.label_to_name)} known faces.")
                return True
        except Exception as e:
            print(f"Could not load existing model: {str(e)}")
        
        return False
    
    def recognize_faces_in_image(self, image_path):
        """
        Recognize faces in a given image.
        
        Args:
            image_path (str): Path to the image to analyze
            
        Returns:
            list: List of dictionaries containing face information
        """
        if not self.is_trained:
            print("Face recognizer is not trained. Please add known faces and retrain.")
            return []
        
        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                return []
            
            # Detect faces
            faces, gray = self.detect_faces(image)
            
            results = []
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100, 100))
                
                # Predict
                label, confidence = self.face_recognizer.predict(face_roi)
                
                # Convert confidence to similarity (lower confidence = better match)
                # OpenCV LBPH returns distance, so we convert to similarity percentage
                similarity = max(0, (100 - confidence) / 100)
                
                name = "Unknown"
                if confidence < 80:  # Threshold for recognition
                    name = self.label_to_name.get(label, "Unknown")
                
                results.append({
                    'name': name,
                    'confidence': similarity,
                    'location': (y, x+w, y+h, x)  # top, right, bottom, left format
                })
            
            return results
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return []
    
    def draw_results_on_image(self, image_path, output_path=None):
        """
        Draw recognition results on the image and save it.
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save annotated image (optional)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return
        
        # Get recognition results
        results = self.recognize_faces_in_image(image_path)
        
        # Draw rectangles and labels
        for result in results:
            top, right, bottom, left = result['location']
            name = result['name']
            confidence = result['confidence']
            
            # Draw rectangle around face
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            
            # Draw label
            label = f"{name} ({confidence:.2f})" if name != "Unknown" else name
            cv2.rectangle(image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(image, label, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Save or display result
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Annotated image saved to: {output_path}")
        else:
            cv2.imshow('Face Recognition Results', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


@click.group()
def cli():
    """OpenCV Face Recognition Application CLI"""
    pass

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output path for annotated image')
def recognize(image_path, output):
    """Recognize faces in an image."""
    system = OpenCVFaceRecognizer()
    
    if not system.is_trained:
        print("No trained model available. Please add face images to the 'known_faces' directory and run 'retrain'.")
        return
    
    print(f"Analyzing image: {image_path}")
    results = system.recognize_faces_in_image(image_path)
    
    if not results:
        print("No faces detected in the image.")
        return
    
    print(f"\nFound {len(results)} face(s):")
    for i, result in enumerate(results, 1):
        name = result['name']
        confidence = result['confidence']
        if name != "Unknown":
            print(f"  {i}. {name} (confidence: {confidence:.2f})")
        else:
            print(f"  {i}. Unknown person")
    
    # Draw results on image
    system.draw_results_on_image(image_path, output)

@cli.command()
def list_known():
    """List all known faces."""
    system = OpenCVFaceRecognizer()
    
    if system.label_to_name:
        print("Known faces:")
        for label, name in system.label_to_name.items():
            print(f"  - {name}")
    else:
        print("No known faces found. Add face images to the 'known_faces' directory and run 'retrain'.")

@cli.command()
def retrain():
    """Retrain the face recognizer with current known faces."""
    system = OpenCVFaceRecognizer()
    
    # Clear existing model
    system.is_trained = False
    system.label_to_name = {}
    
    # Retrain
    if system.train_recognizer():
        print("Retraining completed successfully!")
    else:
        print("Retraining failed. Please check that you have face images in the 'known_faces' directory.")

@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--output-dir', '-o', help='Output directory for annotated images')
def batch_recognize(directory, output_dir):
    """Recognize faces in all images in a directory."""
    system = OpenCVFaceRecognizer()
    
    if not system.is_trained:
        print("No trained model available. Please add face images to the 'known_faces' directory and run 'retrain'.")
        return
    
    directory = Path(directory)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in directory.iterdir() 
                   if f.suffix.lower() in supported_formats]
    
    if not image_files:
        print(f"No image files found in {directory}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    for image_file in image_files:
        print(f"\nProcessing: {image_file.name}")
        results = system.recognize_faces_in_image(str(image_file))
        
        if results:
            for result in results:
                name = result['name']
                confidence = result['confidence']
                if name != "Unknown":
                    print(f"  Found: {name} (confidence: {confidence:.2f})")
                else:
                    print(f"  Found: Unknown person")
            
            # Save annotated image if output directory specified
            if output_dir:
                output_path = output_dir / f"annotated_{image_file.name}"
                system.draw_results_on_image(str(image_file), str(output_path))
        else:
            print("  No faces detected")

if __name__ == '__main__':
    cli()
