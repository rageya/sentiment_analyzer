# 🎭 Sentiment Analyzer — ADL Capstone Assignment 13

A full-stack ML web application that classifies text as **Positive**, **Negative**,
or **Neutral** using TF-IDF + Logistic Regression, served via a FastAPI back-end
and an editorial-style HTML/JS front-end.

---

## 📁 Project Structure

```
sentiment-analyzer/
├── train_model.py          # Step 1 — Train & save the ML model
├── main.py                 # Step 2 — FastAPI back-end (API server)
├── requirements.txt        # Python dependencies
├── render.yaml             # Render.com deployment config
├── model.pkl               # Generated after running train_model.py
└── templates/
    └── index.html          # Front-end UI (HTML + CSS + Vanilla JS)
```

---

## ⚙️ Tech Stack

| Layer      | Technology |
|------------|------------|
| ML Model   | Scikit-Learn — TF-IDF Vectorizer + Logistic Regression |
| Back-End   | FastAPI + Uvicorn |
| Front-End  | HTML5, CSS3, Vanilla JS (Fetch API) |
| Deployment | Render.com (free tier) |

---

## 🚀 Run Locally — Step by Step

### 1 — Set up a virtual environment

```bash
python -m venv venv

# Activate on Mac / Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Train the model (creates model.pkl)

```bash
python train_model.py
```

Expected output:
```
=== Sentiment Analysis Model Evaluation ===

Accuracy : 1.0000

              precision  recall  f1-score ...
    Negative       1.00    1.00      1.00
    Positive       1.00    1.00      1.00
     Neutral       1.00    1.00      1.00

✅  model.pkl saved successfully!

--- Quick sanity check ---
  [Positive 99.2%]  This is absolutely amazing, I love it!
  [Negative 98.5%]  Terrible product, complete waste of money.
  [Neutral  87.3%]  It is okay, nothing special.
```

### 4 — Start the FastAPI server

```bash
uvicorn main:app --reload
```

Expected output:
```
✅  Model loaded successfully.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 5 — Open the app

Navigate to **http://127.0.0.1:8000** in your browser.

Auto-generated API docs: **http://127.0.0.1:8000/docs**

---

## 🔌 API Reference

### `POST /predict`

**Request body (JSON):**
```json
{
  "text": "I absolutely loved this product! Exceeded all expectations."
}
```

**Response:**
```json
{
  "sentiment":    "Positive",
  "emoji":        "😊",
  "confidence":   0.9821,
  "probabilities": {
    "Positive": 0.9821,
    "Negative": 0.0094,
    "Neutral":  0.0085
  },
  "word_count": 9
}
```

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

### `GET /docs`

Interactive Swagger UI — test every endpoint in the browser.

---

## ☁️ Deploy to Render.com (Free, ~2 minutes)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit — Sentiment Analyzer"
git remote add origin https://github.com/<your-username>/sentiment-analyzer.git
git push -u origin main
```

### Step 2 — Create a free account on Render

Go to **https://render.com** and sign up.

### Step 3 — Create a new Web Service

1. **New +** → **Web Service**
2. Connect your GitHub account and select your repository
3. Render auto-reads `render.yaml` — confirm these settings:

| Setting       | Value |
|---------------|-------|
| Environment   | Python |
| Build Command | `pip install -r requirements.txt && python train_model.py` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Instance Type | Free  |

4. Click **Create Web Service**

### Step 4 — Get your public URL

After ~2 min Render gives you a live URL:
```
https://sentiment-analyzer.onrender.com
```

---

## 🏗️ Architecture Diagram

```
[Browser: index.html]
        │
        │  User types text → clicks "Analyse Sentiment"
        ↓
[JavaScript fetch()]
        │
        │  POST /predict  →  { "text": "..." }
        ↓
[FastAPI main.py]
        │
        │  Validates input (Pydantic BaseModel)
        ↓
[Scikit-Learn Pipeline]
   TfidfVectorizer (bigrams, 10k features)
        ↓
   LogisticRegression (multinomial, balanced)
        │
        │  Returns { sentiment, emoji, confidence, probabilities }
        ↓
[JavaScript updates the DOM]
        │
        ↓
[User sees animated result card with probability bars]
```

---

## 📈 Extending the Project

- Replace the sample data with the **IMDB 50k movie reviews** dataset (Kaggle) for much higher accuracy
- Try `SVC`, `RandomForestClassifier`, or upgrade to `transformers` (BERT) for state-of-the-art results
- Add a **history panel** to show all past analyses in the same session
- Add **word highlighting** — colour individual words by their contribution to the prediction
- Swap TF-IDF for **sentence-transformers** embeddings for semantic understanding
- Try deploying on **Hugging Face Spaces** with a `requirements.txt` + `app.py`

---

## 📝 Assignment Checklist

- [x] Machine Learning Model (TF-IDF + Logistic Regression, 3-class)
- [x] Model serialised with `pickle` (`.pkl` file)
- [x] Back-end API (FastAPI, `POST /predict` endpoint with Pydantic validation)
- [x] CORS enabled for cross-origin requests
- [x] Auto-generated Swagger docs at `/docs`
- [x] Front-end HTML/CSS/JS with textarea input
- [x] `fetch()` API connecting front-end to back-end
- [x] Animated probability breakdown bars
- [x] Sample text chips for quick demos
- [x] Deployment config for Render (free tier)
- [x] Accessible via browser on the public web
