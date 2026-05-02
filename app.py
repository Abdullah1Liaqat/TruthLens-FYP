"""
TruthLens – Flask Backend
BERT + LIME + SHAP + NewsAPI + Fact Check + Auth
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity, verify_jwt_in_request
from datetime import datetime, timedelta
import torch
import shap
import json
import requests
import re
from dotenv import load_dotenv
import os
load_dotenv()
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from lime.lime_text import LimeTextExplainer

from models import db, User, Prediction, Explanation

# ==================================================
# APP INIT
# ==================================================
app = Flask(__name__)
CORS(app)

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    raise ValueError("Secret_Key is missing. Check your .env file")
app.config["SQLALCHEMY_DATABASE_URI"]    = "sqlite:///truthlens.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY
app.config["JWT_ACCESS_TOKEN_EXPIRES"]   = timedelta(days=7)

db.init_app(app)
jwt = JWTManager(app)

# ==================================================
# API KEYS  ← Paste your keys here
# ==================================================
NEWS_API_KEY   = os.getenv("NEWS_API_KEY")
FACT_CHECK_KEY = os.getenv("FACT_CHECK_KEY")
if not NEWS_API_KEY or not FACT_CHECK_KEY:
    raise ValueError("API keys are missing. Check your .env file")

# ==================================================
# REGISTER AUTH BLUEPRINT
# ==================================================
from auth import auth_bp
app.register_blueprint(auth_bp)

# ==================================================
# DEVICE + MODEL
# ==================================================
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model/final_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model     = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()
print("BERT model loaded successfully")

# ==================================================
# BERT PREDICT PROBA  (for LIME / SHAP)
# ==================================================
def predict_proba(texts):
    if isinstance(texts, str):
        texts = [texts]
    texts  = list(texts)
    inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                       padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()

# ==================================================
# BERT PREDICTION  (with rule engine)
# ==================================================
def real_predict(text):
    from rule_engine import apply_rules
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs     = F.softmax(outputs.logits, dim=1)
    real_prob = probs[0][1].item()
    fake_prob = probs[0][0].item()
    label     = "REAL" if real_prob > fake_prob else "FAKE"
    confidence= real_prob if label == "REAL" else fake_prob

    final_label, final_conf, rule_score, explanation = apply_rules(text, label, confidence)
    return {
        "prediction": final_label,
        "confidence": round(final_conf, 4),
        "rule_score": rule_score,
        "explanation": explanation
    }

# ==================================================
# HELPERS
# ==================================================
def extract_query(text: str, max_words: int = 10) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", text)
    return " ".join(cleaned.split()[:max_words])

_STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","was","are","were","has","have","had","be","been","being","that",
    "this","it","its","he","she","they","we","i","you","from","by","as",
    "not","no","so","if","do","did","will","would","could","should","may",
    "might","their","there","than","then","which","who","whom","what","when",
    "where","how","all","each","some","any","also","been","into","about",
    "after","before","during","between","over","under","such","both","through"
}

def _keywords(text: str) -> set:
    aliases = {
        "america": "united states", "american": "us",
        "usa": "united states", "u.s.": "united states",
        "pak": "pakistan", "indo": "india",
    }
    t = text.lower()
    for alias, rep in aliases.items():
        t = t.replace(alias, rep)
    words = re.findall(r"[a-zA-Z]+", t)
    return {w for w in words if w not in _STOPWORDS and len(w) >= 4}

# ==================================================
# FACT CHECK API
# ==================================================
def check_factcheck_api(text: str):
    query = extract_query(text, max_words=12)
    url   = (
        "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        f"?query={requests.utils.quote(query)}&key={FACT_CHECK_KEY}"
    )
    try:
        data = requests.get(url, timeout=8).json()
    except Exception as e:
        print("Fact Check API error:", e)
        return False, []

    claims  = data.get("claims", [])
    results = []
    for claim in claims[:4]:
        reviews = claim.get("claimReview", [])
        if not reviews:
            continue
        review    = reviews[0]
        publisher = review.get("publisher", {}).get("name", "Unknown")
        rating    = review.get("textualRating", "Unknown")
        rl        = rating.lower()
        if any(w in rl for w in ["false","fake","incorrect","misleading","pants on fire","mostly false"]):
            verdict = "FAKE"
        elif any(w in rl for w in ["true","correct","accurate","mostly true"]):
            verdict = "REAL"
        else:
            verdict = "UNCERTAIN"
        results.append({
            "publisher":  publisher,
            "rating":     rating,
            "verdict":    verdict,
            "claim_text": claim.get("text", ""),
            "url":        review.get("url", "")
        })
    return (True, results) if results else (False, [])

# ==================================================
# NEWS API  (context only)
# ==================================================
def check_newsapi(text: str):
    MIN_OVERLAP = 2
    MIN_ARTICLES = 1
    query = extract_query(text, max_words=8)
    url   = (
        "https://newsapi.org/v2/everything"
        f"?q={requests.utils.quote(query)}"
        f"&apiKey={NEWS_API_KEY}"
        "&pageSize=10&sortBy=relevancy&language=en"
    )
    try:
        data = requests.get(url, timeout=8).json()
    except Exception as e:
        print("NewsAPI error:", e)
        return False, []

    if data.get("status") != "ok":
        return False, []

    input_kw = _keywords(text)
    matched  = []
    seen     = set()

    for article in data.get("articles", []):
        title   = article.get("title")   or ""
        desc    = article.get("description") or ""
        name    = article.get("source", {}).get("name", "").strip()
        art_url = article.get("url", "")
        overlap = input_kw & _keywords(title + " " + desc)
        if len(overlap) >= MIN_OVERLAP and name and name not in seen:
            seen.add(name)
            matched.append({"name": name, "title": title, "url": art_url})

    return (True, matched[:5]) if len(matched) >= MIN_ARTICLES else (False, [])

# ==================================================
# COMBINED PREDICT
# ==================================================
def combined_predict(text: str) -> dict:
    """
    Priority order (highest to lowest):
      1. NewsAPI        → checked first; if relevant articles found → label REAL
      2. Fact Check API → overrides NewsAPI/model if known claim found
      3. BERT + rule engine → always runs as base verdict
    """
    # BERT always runs first (base verdict)
    bert_result = real_predict(text)
    label       = bert_result["prediction"]
    confidence  = bert_result["confidence"]
    method      = "model"

    # Step 1: NewsAPI checked first
    news_found, news_sources = check_newsapi(text)
    if news_found:
        label      = "REAL"
        confidence = None
        method     = "newsapi"
    else:
        news_sources = []

    # Step 2: Fact Check overrides everything (highest priority)
    fc_found, fc_results = check_factcheck_api(text)
    if fc_found:
        verdicts   = [r["verdict"] for r in fc_results]
        fake_count = verdicts.count("FAKE")
        real_count = verdicts.count("REAL")
        label      = "FAKE" if fake_count > real_count else ("REAL" if real_count > fake_count else "UNCERTAIN")
        method     = "fact_check"
        confidence = None

    return {
        "label":               label,
        "confidence":          confidence,
        "verification_method": method,
        "bert_label":          bert_result["prediction"],
        "bert_confidence":     bert_result["confidence"],
        "fact_checks":         fc_results if fc_found else [],
        "sources":             news_sources,
        "rule_score":          bert_result.get("rule_score"),
        "rule_explanation":    bert_result.get("explanation", [])
    }

# ==================================================
# LIME
# ==================================================
lime_explainer = LimeTextExplainer(class_names=["FAKE", "REAL"], split_expression=r"\W+")

def generate_lime(text):
    exp = lime_explainer.explain_instance(text, predict_proba, num_features=10, num_samples=1000)
    return [{"token": str(w), "value": float(v)} for w, v in exp.as_list()]

# ==================================================
# SHAP
# ==================================================
masker         = shap.maskers.Text(r"\W+")
shap_explainer = shap.Explainer(predict_proba, masker, output_names=["FAKE", "REAL"])

def generate_shap(text):
    sv     = shap_explainer([text])
    tokens = sv.data[0]
    values = sv.values[0]
    if len(values.shape) == 2:
        values = values[:, 1]
    return [{"token": str(t), "value": float(v)} for t, v in zip(tokens, values)
            if t not in ["[CLS]", "[SEP]", "[PAD]"]]

# ==================================================
# HEALTH
# ==================================================
@app.route("/api/health")
def health():
    return jsonify({"status": "OK", "service": "TruthLens API"})

# ==================================================
# PREDICT
# ==================================================
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "text field required"}), 400

    text   = data["text"]
    result = combined_predict(text)

    # Attach user if logged in (optional auth)
    user_id = None
    try:
        verify_jwt_in_request(optional=True)
        uid = get_jwt_identity()
        if uid:
            user_id = int(uid)
    except Exception:
        pass

    db.session.add(Prediction(
        user_id             = user_id,
        text                = text,
        label               = result["label"],
        confidence          = result.get("confidence"),
        verification_method = result["verification_method"]
    ))
    db.session.commit()

    return jsonify({"data": result})

# ==================================================
# LIME
# ==================================================
@app.route("/api/lime", methods=["POST"])
def lime_explain():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "text field required"}), 400
    try:
        explanation = generate_lime(data["text"])
        db.session.add(Explanation(method="lime", explanation=json.dumps(explanation)))
        db.session.commit()
        return jsonify({"data": explanation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==================================================
# SHAP
# ==================================================
@app.route("/api/shap", methods=["POST"])
def shap_explain():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "text field required"}), 400
    try:
        return jsonify({"data": generate_shap(data["text"]), "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

# ==================================================
# HISTORY  (per-user if logged in, else all)
# ==================================================
@app.route("/api/history")
def history():
    try:
        user_id = None
        try:
            verify_jwt_in_request(optional=True)
            uid = get_jwt_identity()
            if uid:
                user_id = int(uid)
        except Exception:
            pass

        q = Prediction.query
        if user_id:
            q = q.filter_by(user_id=user_id)
        records = q.order_by(Prediction.id.desc()).limit(20).all()

        return jsonify([{
            "id":                  r.id,
            "text":                r.text,
            "label":               r.label,
            "confidence":          r.confidence,
            "verification_method": r.verification_method,
            "time":                r.created_at
        } for r in records])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/history/<int:id>", methods=["DELETE"])
def delete_history(id):
    record = Prediction.query.get(id)
    if not record:
        return jsonify({"error": "Not found"}), 404
    db.session.delete(record)
    db.session.commit()
    return jsonify({"message": "deleted"})

# ==================================================
# METRICS
# ==================================================
@app.route("/api/metrics")
def metrics():
    return jsonify({
        "accuracy": 0.91, "precision": 0.89, "recall": 0.90, "f1_score": 0.895,
        "epochs": [
            {"epoch": 1, "accuracy": 0.74, "loss": 0.62},
            {"epoch": 2, "accuracy": 0.83, "loss": 0.44},
            {"epoch": 3, "accuracy": 0.89, "loss": 0.31},
            {"epoch": 4, "accuracy": 0.91, "loss": 0.26}
        ]
    })

# ==================================================
# ABOUT
# ==================================================
@app.route("/api/about")
def about():
    return jsonify({
        "project":     "TruthLens",
        "description": "Fake news detection using BERT with explainable AI",
        "contributors": [
            {"name": "Abdullah Liaqat",  "role": "AI Engineer / Frontend Developer / Integration", "work": "Model training, React UI, dashboards, visualization"},
            {"name": "Muhammad Sajjad",  "role": "Backend Developer",    "work": "RESTful APIs using Flask for model inference"},
            {"name": "Mr Umair Babar",   "role": "Project Supervisor",   "work": "Research guidance and evaluation"}
        ],
        "research": {
            "problem":   "Fake news spreads rapidly online and misleads readers.",
            "gap":       "Many systems focus only on prediction accuracy and ignore explainability.",
            "objective": "Build an accurate and transparent fake news detector using BERT with XAI.",
            "approach":  ["Collected benchmark datasets","Fine-tuned BERT model","Integrated LIME explanations","Integrated SHAP explanations","Built full-stack dashboard"]
        },
        "datasets": ["Kaggle Fake News Dataset", "LIAR Dataset"]
    })

# ==================================================
# INIT DB
# ==================================================
@app.route("/init-db")
def init_db():
    db.create_all()
    return "Database created successfully"

# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=False)