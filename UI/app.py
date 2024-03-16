from flask import Flask, render_template, request, jsonify
from model.model_module import SentimentClassifier, load_model, predict_emotion
import torch
import os

app = Flask(__name__)

project_root = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_root, "model", "phobert_fold3.pth")
model = SentimentClassifier(n_classes=7)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

@app.route('/')
def index():
    return render_template('index.html', script='static/script.js', styles='static/styles.css')

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    comment = request.form['comment']
    emotion = predict_emotion(model, comment)
    return jsonify(emotion)

if __name__ == '__main__':
    app.run(debug=True)
