import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin requests for the GUI

# ==========================================
# 1. DEFINE MODEL ARCHITECTURES
# ==========================================

# --- Custom CNN (Taken exactly from your Notebook) ---
class TrafficSignCNNEnhanced(nn.Module):
    def __init__(self, num_classes=43):
        super(TrafficSignCNNEnhanced, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16 -> 8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- ResNet18 (Recreated based on Notebook Transfer Learning) ---
def get_resnet_model(num_classes=43):
    # Load architecture
    model = models.resnet18(weights=None) # We will load specific weights later
    # Replace the last fully connected layer to match 43 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# ==========================================
# 2. LOAD MODELS
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on Device: {DEVICE}")

# Define Paths (Updated to your D: path)
BASE_PATH = r"Core\models"
CNN_PATH = os.path.join(BASE_PATH, "custom_cnn_finalmodel.pth")
RESNET_PATH = os.path.join(BASE_PATH, "resnet_finalmodel.pth")

# Load CNN
try:
    cnn_model = TrafficSignCNNEnhanced(num_classes=43)
    # Use map_location to handle CPU/GPU saving differences
    cnn_model.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE))
    cnn_model.to(DEVICE)
    cnn_model.eval()
    print("Custom CNN loaded successfully.")
except Exception as e:
    print(f"Error loading CNN: {e}")
    cnn_model = None

# Load ResNet
try:
    resnet_model = get_resnet_model(num_classes=43)
    resnet_model.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
    resnet_model.to(DEVICE)
    resnet_model.eval()
    print("ResNet18 loaded successfully.")
except Exception as e:
    print(f"Error loading ResNet: {e}")
    resnet_model = None

# ==========================================
# 3. PREPROCESSING
# ==========================================
# Using the same stats found in your notebook
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================================
# 4. FLASK ROUTES
# ==========================================

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    model_type = request.form.get('model_type', 'cnn') # 'cnn' or 'resnet'

    try:
        # Open and Preprocess Image
        image = Image.open(file).convert('RGB')
        input_tensor = data_transform(image).unsqueeze(0).to(DEVICE)

        # Select Model
        if model_type == 'resnet':
            if resnet_model is None: return jsonify({'error': 'ResNet model not loaded'}), 500
            model = resnet_model
        else:
            if cnn_model is None: return jsonify({'error': 'CNN model not loaded'}), 500
            model = cnn_model

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        # Return just the ID and confidence (GUI will handle the Name mapping)
        return jsonify({
            'class_id': int(predicted_class.item()),
            'confidence': float(confidence.item()),
            'model_used': model_type
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)