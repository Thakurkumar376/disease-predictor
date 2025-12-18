from flask import Flask, render_template, request, jsonify, session
import joblib
import numpy as np
from difflib import get_close_matches
from flask_cors import CORS
import logging
import uuid
import re

app = Flask(__name__)
app.secret_key = "replace-this-with-a-secure-random-value"
CORS(app)
logging.basicConfig(level=logging.INFO)

# -------------------- Load model bundle --------------------
try:
    bundle = joblib.load("healthcare_model_bundle.pkl")
    clf = bundle.get("decision_tree_model")
    le = bundle.get("label_encoder")
    symptoms_dict = bundle.get("symptoms_dict", {})
    precaution_dict = bundle.get("precaution_dict", {})
    description_dict = bundle.get("description_dict", {})
    logging.info(f"Model loaded: {bundle.get('n_diseases', 0)} diseases, {bundle.get('n_symptoms', 0)} symptoms")
except Exception as e:
    logging.error(f"Failed to load model bundle: {e}")
    clf = le = None
    symptoms_dict = {}
    precaution_dict = {}
    description_dict = {}

# -------------------- Safety rules --------------------
RED_FLAGS = [
    "chest_pain", "breathlessness", "severe_breathing_difficulty", "unconsciousness",
    "sudden_severe_headache", "slurred_speech", "sudden_weakness", "severe_bleeding",
    "blood_in_sputum", "rusty_sputum", "altered_sensorium"
]

RED_FLAG_KEYWORDS = {
    "chest": "chest_pain",
    "shortness of breath": "breathlessness",
    "breath": "breathlessness",
    "faint": "unconsciousness",
    "passed out": "unconsciousness",
    "stroke": "sudden_weakness",
    "slur": "slurred_speech",
    "bleeding": "severe_bleeding",
    "severe headache": "sudden_severe_headache",
    "blood cough": "blood_in_sputum",
}

# -------------------- Symptom matching --------------------
def normalize_symptom_text(text: str) -> str:
    return re.sub(r'[^a-zA-Z_]', '_', text.lower().strip())

def fuzzy_match_symptom(text: str):
    if not symptoms_dict:
        return None
    
    norm = normalize_symptom_text(text)
    
    if norm in symptoms_dict:
        return norm
    
    matches = get_close_matches(norm, symptoms_dict.keys(), n=3, cutoff=0.6)
    if matches:
        return matches[0]
    
    aliases = {
        "fever": "high_fever", "cold": "continuous_sneezing", "headache": "headache",
        "tired": "fatigue", "throat": "throat_irritation", "pain": "abdominal_pain",
        "rash": "skin_rash", "itch": "itching", "nausea": "nausea", "diarrhea": "diarrhoea",
        "dizzy": "dizziness", "tummy": "abdominal_pain", "stomachache": "abdominal_pain",
        "vomiting": "vomiting", "chills": "chills", "joint": "joint_pain", "back": "back_pain",
    }
    
    if norm in aliases:
        alias = aliases[norm]
        if alias in symptoms_dict:
            return alias
    
    return None

def extract_symptoms_from_text(text: str):
    if not text:
        return []
    
    for sep in [",", ";", " and ", "\n", "/", "|"]:
        text = text.replace(sep, ",")
    
    found = []
    chunks = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    
    for chunk in chunks:
        matched = fuzzy_match_symptom(chunk)
        if matched and matched not in found:
            found.append(matched)
            continue
        
        for word in chunk.split():
            matched = fuzzy_match_symptom(word)
            if matched and matched not in found:
                found.append(matched)
    
    return found

def check_red_flags(user_text: str, symptoms: list[str]):
    detected = []
    low = user_text.lower()
    
    for kw, flag in RED_FLAG_KEYWORDS.items():
        if kw in low and flag not in detected:
            detected.append(flag)
    
    for s in symptoms:
        if s in RED_FLAGS and s not in detected:
            detected.append(s)
    
    return detected

def build_input_vector(symptoms: list[str]) -> np.ndarray:
    vec = np.zeros(len(symptoms_dict), dtype=int)
    for s in symptoms:
        idx = symptoms_dict.get(s)
        if idx is not None:
            vec[idx] = 1
    return vec

def model_predict(input_vector: np.ndarray):
    if clf is None or le is None:
        return None, None, None
    
    try:
        probs = clf.predict_proba([input_vector])[0]
        top_idx = np.argmax(probs)
        disease = le.inverse_transform([top_idx])[0]
        confidence = float(probs[top_idx])
        return disease, confidence, probs
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return None, None, None

def init_session():
    session.setdefault("conversation_id", str(uuid.uuid4()))
    session.setdefault("collected_symptoms", [])
    session.setdefault("asked", {})
    session.setdefault("top_predictions", [])
    session.modified = True

# -------------------- Routes --------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    init_session()
    data = request.get_json(silent=True) or {}
    user_msg = data.get("message", "").strip()
    copy_action = data.get("copy_action")

    # Handle copy actions
    if copy_action == "symptoms":
        return jsonify({
            "reply": f"✅ **Symptoms copied!** `{', '.join(session['collected_symptoms'])}`",
            "should_end_session": False,
            "copied_text": ', '.join(session['collected_symptoms'])
        })
    
    if copy_action == "top_prediction":
        if session.get("top_predictions"):
            top_disease = session["top_predictions"][0][0]
            return jsonify({
                "reply": f"✅ **Copied: {top_disease}**",
                "should_end_session": False,
                "copied_text": top_disease
            })

    if not user_msg:
        return jsonify({"reply": "Please enter a message.", "should_end_session": False}), 400

    # Extract NEW symptoms
    new_symptoms = extract_symptoms_from_text(user_msg)
    for s in new_symptoms:
        if s not in session["collected_symptoms"]:
            session["collected_symptoms"].append(s)
    
    if new_symptoms:
        logging.info(f"New symptoms added: {new_symptoms}")

    # Red flags FIRST
    all_symptoms = session["collected_symptoms"]
    flags = check_red_flags(user_msg, all_symptoms)
    if flags:
        return jsonify({
            "reply": f"⚠️ **EMERGENCY** - {', '.join(flags)} detected. Seek medical help NOW! <button onclick=\"copyToClipboard('{', '.join(flags)}')\">📋 Copy</button>",
            "should_end_session": True
        })

    if len(all_symptoms) == 0:
        return jsonify({
            "reply": (
                "Try these common symptoms: <button onclick=\"copyToClipboard('skin rash, itching')\">skin rash, itching</button> | "
                "<button onclick=\"copyToClipboard('continuous sneezing, chills')\">sneezing, chills</button> | "
                "<button onclick=\"copyToClipboard('vomiting, diarrhoea')\">vomiting, diarrhoea</button>"
            ),
            "should_end_session": False
        })

    # Basic questions
    asked = session["asked"]
    if "duration" not in asked:
        asked["duration"] = True
        return jsonify({"reply": "How long? (days)", "should_end_session": False})
    if "severity" not in asked:
        asked["severity"] = True
        return jsonify({"reply": "Severity? (mild/moderate/severe)", "should_end_session": False})
    if "age" not in asked:
        asked["age"] = True
        return jsonify({"reply": "Age?", "should_end_session": False})

    # CONTINUOUS PREDICTION
    vector = build_input_vector(all_symptoms)
    disease, conf, all_probs = model_predict(vector)
    
    if conf and conf >= 0.40:  # FINAL ANSWER
        description = description_dict.get(disease, "No description available.")
        precautions = precaution_dict.get(disease, ["Consult a healthcare professional."])
        
        final_text = f"{disease} - Symptoms: {', '.join(all_symptoms)} (confidence: {conf*100:.1f}%)"
        
        reply = (
            f"**🎉 FINAL DIAGNOSIS**\n"
            f"`{', '.join(all_symptoms)}` → **{disease}** ({conf*100:.1f}%)\n\n"
            f"📋 <button onclick=\"copyToClipboard('{final_text}')\">📋 Copy Result</button>\n\n"
            f"📌 {description}\n"
            f"🛡️ " + "\n".join(f"• {p}" for p in precautions)
        )
        session.clear()
        return jsonify({"reply": reply, "should_end_session": True, "copied_text": final_text})

    # LOW CONFIDENCE → Continue refining
    top3_idx = np.argsort(all_probs)[-3:][::-1]
    top3_diseases = [le.inverse_transform([i])[0] for i in top3_idx]
    top3_confs = [float(all_probs[i]) for i in top3_idx]
    
    session["top_predictions"] = list(zip(top3_diseases, top3_confs))
    
    top3_html = "".join([f"<button onclick=\"copyToClipboard('{d}')\">{d}</button><br>" for d in top3_diseases[:2]])
    
    reply = (
        f"**🔄 Refining... ({len(all_symptoms)} symptoms)**\n"
        f"`{', '.join(all_symptoms)}` <button onclick=\"copyToClipboard('{', '.join(all_symptoms)}')\">📋 Copy</button>\n\n"
        f"**Top predictions:**\n{top3_html}\n"
        f"**Best: {conf*100:.1f}%**\n\n"
        f"💡 **Add more:** nausea, fatigue, joint pain, rash, chills?"
    )
    
    return jsonify({"reply": reply, "should_end_session": False})

@app.route("/api/reset", methods=["POST"])
def reset():
    session.clear()
    init_session()
    return jsonify({"status": "reset"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
