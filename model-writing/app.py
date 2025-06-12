from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib

# ✅ Load model & vectorizer dari file .pkl atau .joblib
model = joblib.load("model-writting.pkl", "rb")  # BUKAN .keras!
vectorizer = joblib.load("models/tokenizer.joblib")
df = pd.read_csv("Dataset-W.csv", encoding='latin1', on_bad_lines='skip')


# ✅ Inisialisasi FastAPI
app = FastAPI()

# ✅ Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Root endpoint
@app.get("/")
async def index():
    return {"message": "API Multiple Choice Model (.pkl) is running."}

# ✅ Endpoint prediksi
@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    question = data.get("question")
    options = data.get("options")
    user_answer = data.get("user_answer")

    if not question or not options:
        return {"error": "Pertanyaan atau opsi tidak ditemukan."}

    # Buat format input ke model
    input_text = question + " " + " ".join([f"{k}: {v}" for k, v in options.items()])
    X = vectorizer.transform([input_text])
    preds = model.predict(X)
    
    pred_idx = int(preds[0])  # misalnya output 0,1,2,3
    pred_letter = chr(pred_idx + 65)  # A, B, C, D
    is_correct = (user_answer.upper() == pred_letter)
    status = "Benar ✅" if is_correct else "Salah ❌"

    # Penjelasan dari Dataset-W.csv
    matched_row = df[df['question'].str.strip() == question.strip()]
    explanation = "Penjelasan tidak tersedia."
    if not matched_row.empty:
        explanation = matched_row.iloc[0][f'label_{pred_letter}']

    return {
        "prediction": pred_letter,
        "status": status,
        "explanation": explanation
    }
