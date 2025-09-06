# Face Recognition Project

A Python-based face recognition system that can identify known faces from images using OpenCV and the face_recognition library.

## Features

- **Face Detection**: Automatically detect faces in images
- **Face Recognition**: Identify known faces with confidence scores
- **Batch Processing**: Process multiple images at once
- **Visual Output**: Generate annotated images with face labels
- **Command Line Interface**: Easy-to-use CLI for all operations

## Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone or download this project** to your local machine

2. **Navigate to the project directory**:
   ```bash
   cd face_recognition_project
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: Installing `dlib` might take some time as it needs to compile. If you encounter issues:
   - On Windows: Consider installing Visual Studio Build Tools
   - On macOS: Install Xcode command line tools with `xcode-select --install`
   - On Linux: Install cmake and build-essential

## Usage

### 1. Add Known Faces

Before you can recognize faces, you need to add reference images of people you want to identify:

1. Place face images in the `known_faces/` directory
2. Name each file with the person's name (e.g., `john_doe.jpg`, `jane_smith.png`)
3. Use clear, front-facing photos with good lighting
4. One face per image works best

**Example structure**:
```
known_faces/
├── alice.jpg
├── bob.png
├── charlie.jpeg
└── diana.bmp
```

### 2. Recognize Faces in a Single Image

```bash
python face_recognition_app.py recognize path/to/your/image.jpg
```

**Options**:
- `--output` or `-o`: Save annotated image to specified path
- `--tolerance` or `-t`: Set recognition tolerance (0.0-1.0, default: 0.6)

**Examples**:
```bash
# Basic recognition
python face_recognition_app.py recognize test_images/group_photo.jpg

# Save annotated result
python face_recognition_app.py recognize test_images/group_photo.jpg -o results/annotated_group.jpg

# Use stricter tolerance
python face_recognition_app.py recognize test_images/group_photo.jpg -t 0.4
```

### 3. List Known Faces

```bash
python face_recognition_app.py list-known
```

### 4. Batch Process Multiple Images

```bash
python face_recognition_app.py batch-recognize path/to/image/directory
```

**Options**:
- `--output-dir` or `-o`: Directory to save annotated images

**Example**:
```bash
python face_recognition_app.py batch-recognize test_images/ -o results/
```

## Understanding Results

### Confidence Scores
- **0.0 - 0.4**: Low confidence (might be incorrect)
- **0.4 - 0.6**: Medium confidence (likely correct)
- **0.6 - 1.0**: High confidence (very likely correct)

### Tolerance Settings
- **Lower tolerance (0.3-0.5)**: More strict, fewer false positives
- **Higher tolerance (0.6-0.8)**: More lenient, may include more matches

## Project Structure

```
face_recognition_project/
├── face_recognition_app.py    # Main application
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── known_faces/              # Directory for reference face images
└── test_images/              # Directory for images to analyze
```

## Troubleshooting

### Common Issues

1. **"No module named 'face_recognition'"**
   - Run: `pip install -r requirements.txt`

2. **"No faces detected"**
   - Ensure the image contains clear, visible faces
   - Try images with better lighting or higher resolution

3. **"No known faces loaded"**
   - Add face images to the `known_faces/` directory
   - Ensure images contain detectable faces

4. **Poor recognition accuracy**
   - Use high-quality reference images
   - Adjust tolerance settings
   - Ensure good lighting in both reference and test images

### Performance Tips

- Use images with resolution between 300x300 and 1000x1000 pixels for best performance
- Ensure faces are at least 100x100 pixels in the image
- Use well-lit, front-facing photos for reference images
- Avoid blurry or heavily compressed images

## Dependencies

- **opencv-python**: Computer vision library for image processing
- **face-recognition**: Face recognition library built on dlib
- **numpy**: Numerical computing library
- **Pillow**: Python Imaging Library
- **click**: Command line interface creation toolkit
- **dlib**: Machine learning library (required by face-recognition)

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this face recognition system.
# face_recognition
# face_recognition
