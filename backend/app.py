from flask import Flask, jsonify
from model.train import train_model

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status": "TruthLens backend running"})

@app.route("/train")
def train():
    train_model(data_path="datasets/liar")
    return jsonify({"message": "Training completed"})

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, jsonify
from model.train import train_model

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status": "TruthLens backend running"})

@app.route("/train")
def train():
    train_model(data_path="datasets/liar")
    return jsonify({"message": "Training completed"})

if __name__ == "__main__":
    app.run(debug=True)
