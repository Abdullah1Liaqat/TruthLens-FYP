# 📌 TruthLens – Fake News Detection System

TruthLens is a fake news detection system based on Transformer models (**RoBERTa**) combined with Explainable AI techniques such as **SHAP** and **LIME**, along with real-time fact verification using external APIs.

---

## 🧠 Core Features

* Fake news classification using Transformer model (**RoBERTa**)
* Real-time news verification using external APIs
* Text preprocessing and dataset handling
* Explainable predictions (SHAP / LIME)
* REST-based Flask backend
* Prototype React frontend

---

## 🛠️ Technologies Used

* Python
* Flask
* React
* RoBERTa (Transformer model)
* SHAP & LIME (Explainable AI)
* SQL

##

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/your-username/TruthLens.git
cd TruthLens
```

---

## 🐍 Backend Setup (Flask)

### Install dependencies

```bash
pip install -r requirements.txt
```

---

### Create `.env` file inside backend/

```env
NEWS_API_KEY=your_news_api_key
FACT_CHECK_KEY=your_fact_check_api_key
JWT_SECRET_KEY=your_secret_key
```

---

### Run backend

```bash
python app.py
```

Backend runs on:

```
http://localhost:5000
```

---

## ⚛️ Frontend Setup (React)

```bash
cd frontend
npm install
npm start
```

Frontend runs on:

```
http://localhost:3000
```

---

## 🔑 API Keys Setup

To run the system, you need API keys:

### 1. News API

Get key from:

* https://newsapi.org/

### 2. Fact Check API

Use:

* Google Fact Check Tools API
* https://developers.google.com/fact-check/tools/api

---

## ⚠️ Important Notes

* Never upload `.env` file to GitHub
* Use `.env.example` for reference
* Model used: **RoBERTa (not BERT)**
* System supports real-time verification using APIs

---

## 👨‍🎓 Team

* Abdullah Liaqat – Project Lead / AI & Model Development
* Muhammad Sajjad – Dataset & Backend Support

---

## 👨‍🏫 Supervisor

* Sir Umair Babar
