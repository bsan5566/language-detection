from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import os

app = Flask(__name__, static_folder='../frontend')
CORS(app)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
model = joblib.load(model_path)

@app.route("/")
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    prediction = model.predict([text])[0]
    return jsonify({"language": prediction})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
