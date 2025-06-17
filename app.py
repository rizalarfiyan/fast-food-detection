from flask import Flask, render_template, jsonify
from datetime import datetime
import random

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    year = datetime.now().year
    return render_template("index.html", year=year)

@app.route('/detection', methods=['POST'])
def detection():
    class_names = [
        "Burger",
        "Donat",
        "Hot dog",
        "Pizza",
        "Sandwich",
        "Baked Potato",
        "Crispy Chicken",
        "Fries",
        "Taco",
        "Taquito",
    ]

    confidences = [random.randint(1, 100) for _ in class_names]
    total_confidence = sum(confidences)
    normalized_confidences = [round((c / total_confidence) * 100, 2) for c in confidences]

    data = [
        {"name": class_name, "confidence": confidence}
        for class_name, confidence in zip(class_names, normalized_confidences)
    ]

    data.sort(key=lambda x: x["confidence"], reverse=True)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
