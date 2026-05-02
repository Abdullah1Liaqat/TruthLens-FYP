"""
TruthLens – Auth Routes
Signup, Login, Me, Logout
"""

from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
    get_jwt_identity
)
from models import db, User

auth_bp = Blueprint("auth", __name__)

# ==================================================
# SIGNUP
# ==================================================
@auth_bp.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json()

    username = (data.get("username") or "").strip()
    email    = (data.get("email")    or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not username or not email or not password:
        return jsonify({"error": "username, email and password are required"}), 400

    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 409

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already taken"}), 409

    hashed = generate_password_hash(password)
    user   = User(username=username, email=email, password=hashed)

    db.session.add(user)
    db.session.commit()

    token = create_access_token(identity=str(user.id))

    return jsonify({
        "message": "Account created successfully",
        "token":   token,
        "user": {
            "id":       user.id,
            "username": user.username,
            "email":    user.email
        }
    }), 201


# ==================================================
# LOGIN
# ==================================================
@auth_bp.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()

    email    = (data.get("email")    or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    user = User.query.filter_by(email=email).first()

    if not user or not check_password_hash(user.password, password):
        return jsonify({"error": "Invalid email or password"}), 401

    token = create_access_token(identity=str(user.id))

    return jsonify({
        "message": "Login successful",
        "token":   token,
        "user": {
            "id":       user.id,
            "username": user.username,
            "email":    user.email
        }
    })


# ==================================================
# ME  (get current user from token)
# ==================================================
@auth_bp.route("/api/auth/me", methods=["GET"])
@jwt_required()
def me():
    user_id = int(get_jwt_identity())
    user    = User.query.get(user_id)

    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({
        "id":       user.id,
        "username": user.username,
        "email":    user.email
    })