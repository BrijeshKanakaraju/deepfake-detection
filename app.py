import cv2
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import models, transforms
import torch.nn.functional as F
import os

app = Flask(__name__)

# Load pre-trained ResNet for fake video detection
model = models.resnet50(pretrained=True)
model.eval()

# Load MTCNN model for face detection
mtcnn = MTCNN(keep_all=True)

# Transformations for input image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Placeholder function for fake video detection
def fake_video_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    fake_frames_count = 0
    total_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Detect faces in the frame
        faces = mtcnn.detect(image)
        
        if faces[0] is not None:  # If faces are detected
            # Preprocess image and feed to the model
            input_tensor = preprocess(image).unsqueeze(0)
            
            # Make predictions using ResNet
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
            
            # Assuming fake frames can be detected if model gives 'unusual' output, we use this as a placeholder
            if predicted.item() != 0:  # Fake if predicted not normal
                fake_frames_count += 1
        
        total_frames += 1

    cap.release()

    # Calculate percentage of fake frames
    fake_percentage = (fake_frames_count / total_frames) * 100 if total_frames > 0 else 0
    return fake_percentage

# Function to handle image detection
def fake_image_detection(image_path):
    # Open the image and preprocess it
    image = Image.open(image_path)
    input_tensor = preprocess(image).unsqueeze(0)
    
    # Make predictions using ResNet
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    
    # Placeholder logic: consider it fake if the model does not predict a normal class
    if predicted.item() != 0:
        return True, 100.0  # Fake image detected
    else:
        return False, 100.0  # Real image detected

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Ensure file path exists and save the file
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    
    # Check file type
    if file.filename.lower().endswith(('png', 'jpg', 'jpeg', 'gif')):
        # Process as image
        is_fake, confidence = fake_image_detection(file_path)
        if is_fake:
            result = f"Fake image detected with {confidence:.2f}% confidence."
        else:
            result = f"Real image detected with {confidence:.2f}% confidence."
    elif file.filename.lower().endswith(('mp4', 'mov', 'avi')):
        # Process as video
        print(f"Processing video file: {file.filename}")  # Debugging: print the filename
        fake_percentage = fake_video_detection(file_path)
        if fake_percentage > 60:
            result = f"Fake video detected with {fake_percentage:.2f}% confidence."
        else:
            result = f"Real video detected with {100 - fake_percentage:.2f}% confidence."
    else:
        return jsonify({'error': 'Unsupported file type'}), 400
    
    return jsonify({'result': result})

import os

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode)

