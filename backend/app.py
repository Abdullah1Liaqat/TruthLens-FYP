"""
TruthLens – Flask Backend ( API FOR ALL SCREENS)
=================================================
(No transformers / torch / multiprocessing / debug tools)

This backend CONNECTS to ALL React screens:
 Analyze  -> /api/predict
 Explain  -> /api/shap
 Metrics  -> /api/metrics
 About    -> /api/about
 Health   -> /api/health

NOTE:
• Model is MOCKED  for now due to sandbox limits
• API contract is IDENTICAL to real BERT system

"""

# ==================================================
# IMPORTS (SAFE)
# ==================================================
from flask import Flask, request, jsonify
from flask_cors import CORS

# ==================================================
# APP INIT
# ==================================================
app = Flask(__name__)
CORS(app)

# ==================================================
# MOCK MODEL (DETERMINISTIC)
# ==================================================

def predict_text(text: str):
    text = text.lower()

    fake_words = ["fake", "hoax", "rumor", "shocking", "secret"]
    real_words = ["official", "confirmed", "government", "report", "verified"]

    fake_score = sum(w in text for w in fake_words)
    real_score = sum(w in text for w in real_words)

    if fake_score > real_score:
        return "FAKE", round(0.65 + fake_score * 0.07, 2)
    return "REAL", round(0.65 + real_score * 0.07, 2)

# ==================================================
# HEALTH CHECK (OPTIONAL)
# ==================================================
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "OK", "service": "TruthLens API"})

# ==================================================
# ANALYZE SCREEN API
# ==================================================
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({"error": "text field required"}), 400

    label, confidence = predict_text(data['text'])

    return jsonify({
        "data": {
            "label": label,
            "confidence": confidence
        }
    })

# ==================================================
# EXPLAIN SCREEN (SHAP ONLY)
# ==================================================
@app.route('/api/shap', methods=['POST'])
def shap_explain():
    # Mock SHAP output (token-level contributions)
    return jsonify({
        "data": [
            {"token": "secret", "value": 0.32},
            {"token": "shocking", "value": 0.21},
            {"token": "official", "value": -0.27},
            {"token": "verified", "value": -0.18}
        ]
    })

# ==================================================
# METRICS SCREEN API
# ==================================================
@app.route('/api/metrics', methods=['GET'])
def metrics():
    return jsonify({
        "accuracy": 0.72,
        "precision": 0.74,
        "recall": 0.70,
        "f1_score": 0.72,
        "epochs": [
            {"epoch": 1, "accuracy": 0.55, "loss": 0.88},
            {"epoch": 2, "accuracy": 0.63, "loss": 0.71},
            {"epoch": 3, "accuracy": 0.69, "loss": 0.58},
            {"epoch": 4, "accuracy": 0.72, "loss": 0.49}
        ]
    })

# ==================================================
# ABOUT SCREEN API
# ==================================================
@app.route('/api/about', methods=['GET'])
def about():
    return jsonify({
        "project": "TruthLens",
        "description": "Fake news detection and explainability using transformer models",
        "model": "BERT (mocked in sandbox)",
        "frontend": "React + Tailwind",
        "backend": "Flask REST API",
        "explainability": ["SHAP", "LIME"],
        "academic": "Final Year Project"
    })

# ==================================================
# RUN (SAFE MODE)
# ==================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

# ==================================================
# MANUAL TEST CASES 
# ==================================================
# 1) GET  /api/health
# 2) POST /api/predict  {"text": "Government confirmed official report"}
# 3) POST /api/shap
# 4) GET  /api/metrics
# 5) GET  /api/about
