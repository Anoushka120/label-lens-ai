import re
import os
import joblib
from openai import OpenAI

def verdict_from_score(score):
    if score >= 70:
        return "✅ Good to Buy"
    elif score >= 50:
        return "⚠️ Think Twice"
    return "❌ Avoid"

def analyze_rules(text):
    t = text.lower()
    score = 100
    reasons = []

    flags = {
        "hydrogenated": 20,
        "high fructose corn syrup": 20,
        "corn syrup": 12,
        "maltodextrin": 10,
        "artificial": 8,
        "sodium benzoate": 10
    }

    for k, v in flags.items():
        if k in t:
            score -= v
            reasons.append(f"Contains {k}")

    score = max(0, score)
    return verdict_from_score(score), score, reasons or ["No major red flags"]

def analyze_openai(text):
    client = OpenAI()
    prompt = f"""Analyze this food label and return verdict, score (0-100) and reasons.
{text}"""

    res = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    output = res.output_text
    return "⚠️ AI Review", 65, [output[:300]]

def analyze_ml(text):
    if not os.path.exists("model.joblib"):
        return "⚠️ Model Missing", 50, ["Run train_model.py first"]

    model = joblib.load("model.joblib")
    proba = model.predict_proba([text])[0][1]
    score = int(proba * 100)
    return verdict_from_score(score), score, ["ML-based prediction"]
