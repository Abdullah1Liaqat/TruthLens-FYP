from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(80),  unique=True, nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password   = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Prediction(db.Model):
    id                  = db.Column(db.Integer, primary_key=True)
    user_id             = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    text                = db.Column(db.Text, nullable=False)
    label               = db.Column(db.String(10), nullable=False)
    confidence          = db.Column(db.Float, nullable=True)
    verification_method = db.Column(db.String(20), default="model")
    created_at          = db.Column(db.DateTime, default=datetime.utcnow)

class Explanation(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.Integer)
    method        = db.Column(db.String(10))
    explanation   = db.Column(db.Text)