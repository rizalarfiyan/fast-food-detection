from flask import Flask, render_template, jsonify
from datetime import datetime
import io
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)
class_names = ['Baked Potato', 'Burger', 'Crispy Chicken', 'Donut', 'Fries', 'Hot Dog', 'Pizza', 'Sandwich', 'Taco', 'Taquito']

model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('models/model.pth', map_location=device))
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(226),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return preprocess(image).unsqueeze(0)

@app.route('/', methods=['GET'])
def index():
    year = datetime.now().year
    return render_template("index.html", year=year)

@app.route('/detection', methods=['POST'])
def detection():
    if 'image' not in request.files:
        return jsonify({"error": "File tidak ditemukan"}), 400
    
    image = request.files['image']
    
    if image.filename == '':
        return jsonify({"error": "Tidak ada image yang dipilih"}), 400

    try:
        img_bytes = image.read()
        tensor = transform_image(img_bytes)
        tensor = tensor.to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        results = []
        for i, prob in enumerate(probabilities):
            confidence = float(prob.item()) * 100
            class_name = class_names[i]
            results.append({"name": class_name, "confidence": confidence})

        results.sort(key=lambda x: x['confidence'], reverse=True)
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
