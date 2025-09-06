import os
import cv2
import numpy as np
from typing import List, Tuple

# Initialize LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to load and train known faces
def train_known_faces(known_faces_dir: str) -> Tuple[cv2.face.LBPHFaceRecognizer, List[str]]:
    """
    Loads and trains the LBPH recognizer with images from known_faces directory.
    
    Args:
        known_faces_dir (str): Path to directory with known face images.
    
    Returns:
        Tuple: Trained recognizer and list of label names.
    """
    faces = []
    labels = []
    label_names = []
    
    for idx, filename in enumerate(os.listdir(known_faces_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            label = os.path.splitext(filename)[0].capitalize()
            image_path = os.path.join(known_faces_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Could not load {filename}. Skipping.")
                continue
            faces.append(image)
            labels.append(idx)
            label_names.append(label)
    
    if not faces:
        raise ValueError("No valid images found in known_faces directory.")
    
    recognizer.train(faces, np.array(labels))
    return recognizer, label_names

# Function to recognize faces in a test image
def recognize_faces(test_image_path: str, recognizer: cv2.face.LBPHFaceRecognizer, label_names: List[str]) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """
    Detects and recognizes faces in the test image.
    
    Args:
        test_image_path (str): Path to the test image.
        recognizer: Trained LBPH recognizer.
        label_names (List[str]): List of names corresponding to label indices.
    
    Returns:
        List[Tuple[str, Tuple[int, int, int, int]]]: List of (label, bounding_box) for recognized faces.
    """
    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise ValueError("Failed to load Haar Cascade classifier.")
    
    # Load and preprocess test image
    image = cv2.imread(test_image_path)
    if image is None:
        raise ValueError(f"Could not load test image: {test_image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    results = []
    
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        # Recognize face
        label_idx, confidence = recognizer.predict(face_roi)
        # Use confidence threshold (lower is better; adjust as needed)
        label = label_names[label_idx] if confidence < 100 else "Unknown"
        # Store results as (label, (top, right, bottom, left)) for consistency
        results.append((label, (y, x+w, y+h, x)))
    
    return results

# Function to draw bounding boxes and labels
def draw_results(image_path: str, results: List[Tuple[str, Tuple[int, int, int, int]]], output_path: str):
    """
    Draws bounding boxes and labels on the image and saves the output.
    
    Args:
        image_path (str): Path to the original test image.
        results (List): List of (label, bounding_box) tuples.
        output_path (str): Path to save the annotated image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image for drawing: {image_path}")
    
    for label, (top, right, bottom, left) in results:
        # Draw rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        # Draw label background and text
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
    
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved to: {output_path}")

# Main execution
if __name__ == "__main__":
    # Paths
    known_faces_dir = "known_faces"
    test_image_path = "known_faces/torie.jpg"  # Using existing image for testing
    output_image_path = "output_annotated.jpg"
    
    try:
        # Train the recognizer
        recognizer, label_names = train_known_faces(known_faces_dir)
        print(f"Trained with {len(label_names)} known faces.")
        
        # Recognize faces in test image
        results = recognize_faces(test_image_path, recognizer, label_names)
        
        # Print results
        if results:
            print("Recognized faces:")
            for label, box in results:
                print(f"- {label} at bounding box {box}")
        else:
            print("No faces detected in the test image.")
        
        # Draw and save annotated image
        draw_results(test_image_path, results, output_image_path)
    
    except Exception as e:
        print(f"Error: {str(e)}")