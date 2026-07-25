import os
import re
import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
MODEL_DIR = Path(os.getenv("SAFEFACTORY_MODEL_DIR", ROOT / "backend_analysis/SafeFactory/Model"))
DATA_DIR = Path(os.getenv("SAFEFACTORY_DATA_DIR", ROOT / "backend_analysis/SafeFactory/data"))

# ── load models + knowledge base ─────────────────────────────────────────
binary_pkg = joblib.load(MODEL_DIR / "binary_failure_model.pkl")
multi_pkg = joblib.load(MODEL_DIR / "multiclass_failure_model.pkl")
risk_pkg = joblib.load(MODEL_DIR / "risk_score_model.pkl")
priority_pkg = joblib.load(MODEL_DIR / "maintenance_priority_model.pkl")

BINARY_MODEL = binary_pkg["model"]
BINARY_THRESHOLD = binary_pkg.get("threshold", 0.5)
BINARY_FEATURES = binary_pkg["features"]

MULTI_MODEL = multi_pkg["model"]
MULTI_ENCODER = multi_pkg["label_encoder"]
MULTI_FEATURES = multi_pkg["features"]

RISK_MODEL = risk_pkg["model"]
RISK_FEATURES = risk_pkg["feature_columns"]

PRIORITY_MODEL = priority_pkg["model"]
PRIORITY_FEATURES = priority_pkg["features"]

TYPE_MAP = {"L": 0, "M": 1, "H": 2}
EQUIPMENT_TYPES = ["Cable", "CircuitBreaker", "Relay", "Switch", "Transformer"]

CHAT_FIELDS = [
    "Type", "Air temperature K", "Process temperature K",
    "Rotational speed rpm", "Torque Nm", "Tool wear min",
]
FIELD_PROMPTS = {
    "Type": "Machine type (L / M / H)",
    "Air temperature K": "Air temperature (K)",
    "Process temperature K": "Process temperature (K)",
    "Rotational speed rpm": "Rotational speed (rpm)",
    "Torque Nm": "Torque (Nm)",
    "Tool wear min": "Tool wear (min)",
}

kb_df = pd.read_csv(DATA_DIR / "failure_knowledge_base.csv")
KB = {row["Failure_Type"]: row.to_dict() for _, row in kb_df.iterrows()}

# ── simple TF-IDF retriever over the knowledge base ─────────────────────
def kb_row_text(row):
    return (
        f"Failure Type: {row['Failure_Type']}\n"
        f"Description: {row['Description']}\n"
        f"Symptoms: {row['Symptoms']}\n"
        f"Typical Indicators: {row['Typical_Indicators']}\n"
        f"Root Cause: {row['Root_Cause']}\n"
        f"Recommendations: {row['Recommendation_1']}. {row['Recommendation_2']}. {row['Recommendation_3']}\n"
        f"Prevention: {row['Prevention']}\n"
        f"FAQ: {row['FAQ']}"
    )

_vectorizer = TfidfVectorizer(stop_words="english")
_kb_matrix = _vectorizer.fit_transform(kb_df.apply(kb_row_text, axis=1))

def retrieve_kb(query, top_k=1):
    sims = cosine_similarity(_vectorizer.transform([query]), _kb_matrix).flatten()
    idxs = sims.argsort()[::-1][:top_k]
    return [kb_df.iloc[i].to_dict() for i in idxs]

# ── LLM helper ────────────────────────────────────────────────────────────
def has_openai():
    return bool(os.getenv("OPENAI_API_KEY"))

def call_llm(prompt, max_tokens=400):
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def extract_sensor_readings(text):
    if not has_openai():
        return None
    prompt = f"""
Extract machine sensor readings from the text below.

Recognize synonyms:
Machine Type: L/l/low/light, M/m/mid/medium, H/h/high/heavy
Air Temperature: air temp, ambient temp
Process Temperature: process temp
Rotational Speed: speed, rpm
Torque: torque
Tool Wear: wear, tool wear

Return ONLY valid JSON, no commentary, no markdown fences:
{{
  "Type": "L|M|H|null",
  "Air temperature K": number|null,
  "Process temperature K": number|null,
  "Rotational speed rpm": number|null,
  "Torque Nm": number|null,
  "Tool wear min": number|null
}}

Text:
{text}
"""
    try:
        raw = call_llm(prompt, max_tokens=300)
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception:
        return None

def risk_level(score):
    if score < 0.35:
        return "Low"
    if score < 0.60:
        return "Medium"
    if score < 0.80:
        return "High"
    return "Critical"

# ── per-session chat state ───────────────────────────────────────────────
# each session tracks the 6 sensor fields, plus whether we've already
# given the customer the full explanation once (so follow-ups can be short
# and specific instead of repeating everything).
sessions: dict[str, dict] = {}

def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {f: None for f in CHAT_FIELDS}
        sessions[session_id]["_explained"] = False
    return sessions[session_id]

# ── shared prediction logic ──────────────────────────────────────────────
def normalize_type(value):
    """Map any reasonable form of machine type (L/M/H, lowercase, with
    spaces, etc.) to the 0/1/2 the models were trained on."""
    if isinstance(value, (int, float)):
        return int(value)
    key = str(value).strip().upper()[:1]
    if key not in TYPE_MAP:
        raise ValueError(f"Unrecognized machine type: {value!r}")
    return TYPE_MAP[key]


def predict_failure(readings):
    row = readings.copy()
    row["Type"] = normalize_type(row["Type"])

    X_bin = pd.DataFrame([row])[BINARY_FEATURES]
    fail_prob = float(BINARY_MODEL.predict_proba(X_bin)[0, 1])
    will_fail = fail_prob >= BINARY_THRESHOLD

    if not will_fail:
        return {"will_fail": False, "failure_type": None, "fail_prob": fail_prob}

    X_multi = pd.DataFrame([row])[MULTI_FEATURES]
    pred_idx = MULTI_MODEL.predict(X_multi)[0]
    failure_type = MULTI_ENCODER.inverse_transform([pred_idx])[0]
    return {"will_fail": True, "failure_type": failure_type, "fail_prob": fail_prob}

def explain_result(readings, prediction, question="", first_time=True):
    """
    Answer the customer's actual question in plain language, grounded in
    the prediction + knowledge base. On the FIRST reply for a session we
    give a short overall picture; after that we answer only what was asked
    instead of repeating the full rundown every turn.
    """
    if not prediction["will_fail"]:
        status, issue = "Healthy", None
        match = (retrieve_kb("No Failure") or [{}])[0]
    else:
        status, issue = "Attention Recommended", prediction["failure_type"]
        match = (retrieve_kb(f"failure type {issue}", top_k=1) or [{}])[0]

    if has_openai():
        if first_time:
            scope = (
                "This is the first time you're telling the customer about this "
                "result, so give a brief, friendly overview: what the status means, "
                "and whether they should do anything about it. Keep it tight — a "
                "few sentences, not a full report."
            )
        else:
            scope = (
                "The customer already knows the machine's status. Answer ONLY "
                "their specific question below, directly and conversationally. "
                "Do not repeat the full status, cost, cause, and prevention "
                "checklist unless they actually asked about that topic."
            )

        prompt = f"""
You are a customer-friendly machine health assistant.
The machine learning model has already made this prediction — never change it,
and never invent another problem.

Customer question: {question or "What is going on with my machine?"}
Machine status: {status}
Predicted issue: {issue or "none"}
Chance of a problem: {prediction['fail_prob']:.2%}
Machine information: {json.dumps(readings, indent=2)}
Knowledge base: {json.dumps(match, indent=2)}

{scope}

Avoid jargon. Do not mention machine learning or raw sensor values.
"""
        try:
            return call_llm(prompt, max_tokens=250 if not first_time else 300)
        except Exception:
            pass

    # fallback if no LLM available
    if not prediction["will_fail"]:
        return (
            f"Your machine appears to be operating normally "
            f"(estimated chance of an issue: {prediction['fail_prob']:.0%}). "
            "No action needed — keep up with routine maintenance."
        )
    recs = [match.get("Recommendation_1"), match.get("Recommendation_2"), match.get("Recommendation_3")]
    recs = [r for r in recs if r]
    cost = match.get("Estimated_Cost_USD")
    return (
        f"Predicted issue: {issue} (confidence: {prediction['fail_prob']:.0%})\n"
        f"Likely cause: {match.get('Root_Cause', 'unavailable')}\n"
        f"Priority: {match.get('Priority', 'unavailable')}\n"
        f"Estimated repair cost: {'$' + format(cost, ',.0f') if cost is not None else 'unavailable'}\n"
        "Recommended actions:\n" + "\n".join(f" - {r}" for r in recs)
    )

def answer_kb_question(question, kb_row):
    """Answer a free-text maintenance question using one KB entry."""
    if not kb_row:
        return "I couldn't find information on that."

    if not has_openai():
        return (
            f"{kb_row.get('Description', '')} "
            f"Root cause: {kb_row.get('Root_Cause', '')}. "
            f"Recommendations: {kb_row.get('Recommendation_1', '')}; "
            f"{kb_row.get('Recommendation_2', '')}; {kb_row.get('Recommendation_3', '')}."
        )

    prompt = f"""You are a maintenance knowledge assistant for an industrial facility.

Failure type: {kb_row.get('Failure_Type', '')}
Description: {kb_row.get('Description', '')}
Root Cause: {kb_row.get('Root_Cause', '')}
Symptoms: {kb_row.get('Symptoms', '')}
Severity: {kb_row.get('Severity', '')}
Priority: {kb_row.get('Priority', '')}
Estimated cost: ${int(kb_row.get('Estimated_Cost_USD', 0)):,}
Estimated downtime: {kb_row.get('Downtime_Hours', 0)} hours
Recommendations: {kb_row.get('Recommendation_1', '')}; {kb_row.get('Recommendation_2', '')}; {kb_row.get('Recommendation_3', '')}
Prevention: {kb_row.get('Prevention', '')}
FAQ: {kb_row.get('FAQ', '')}

Technician's question: {question}

Answer ONLY what the technician asked, using the information above. Do not
cover every topic (cost, cause, prevention, etc.) unless they asked about it.
Do not invent new information. Do not mention machine learning or AI.
Keep under 100 words.
"""
    try:
        return call_llm(prompt, max_tokens=250)
    except Exception:
        return (
            f"{kb_row.get('Description', '')} "
            f"Root cause: {kb_row.get('Root_Cause', '')}."
        )

# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(title="SafeFactory Inference & Copilot API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class PredictRequest(BaseModel):
    machine_type: str
    air_temp_k: float
    process_temp_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float

class RiskRequest(BaseModel):
    location_lat: float
    location_long: float
    ambient_temperature_c: float
    humidity_percent: float
    wind_speed_mps: float
    precipitation_mm: float
    load_current_a: float
    voltage_kv: float
    smart_meter_gasp_signal: float
    vibration_level_g: float
    insulation_resistance_mohm: float
    historical_failures_count: float
    time_since_last_failure_days: float
    equipment_type: str

class PriorityRequest(BaseModel):
    temp_c: float
    vibration_mm_s: float
    pressure_bar: float
    acoustic_db: float
    inspection_duration_min: float
    downtime_cost_usd: float
    technician_availability_pct: float
    risk_score: float

class AskRequest(BaseModel):
    question: str
    failure_type: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/api/predict")
def predict(req: PredictRequest):
    readings = {
        "Type": req.machine_type.upper(),
        "Air temperature K": req.air_temp_k,
        "Process temperature K": req.process_temp_k,
        "Rotational speed rpm": req.rotational_speed_rpm,
        "Torque Nm": req.torque_nm,
        "Tool wear min": req.tool_wear_min,
    }
    prediction = predict_failure(readings)
    kb_entry = KB.get(prediction["failure_type"]) if prediction["will_fail"] else None
    return {
        "will_fail": prediction["will_fail"],
        "fail_prob": round(prediction["fail_prob"], 4),
        "threshold": BINARY_THRESHOLD,
        "failure_type": prediction["failure_type"],
        "kb": kb_entry,
    }

@app.post("/api/risk")
def risk(req: RiskRequest):
    row = {
        "location_lat": req.location_lat,
        "location_long": req.location_long,
        "ambient_temperature_C": req.ambient_temperature_c,
        "humidity_percent": req.humidity_percent,
        "wind_speed_mps": req.wind_speed_mps,
        "precipitation_mm": req.precipitation_mm,
        "load_current_A": req.load_current_a,
        "voltage_kV": req.voltage_kv,
        "smart_meter_gasp_signal": req.smart_meter_gasp_signal,
        "vibration_level_g": req.vibration_level_g,
        "insulation_resistance_MOhm": req.insulation_resistance_mohm,
        "historical_failures_count": req.historical_failures_count,
        "time_since_last_failure_days": req.time_since_last_failure_days,
    }
    for et in EQUIPMENT_TYPES:
        row[f"equipment_type_{et}"] = int(req.equipment_type == et)
    score = float(RISK_MODEL.predict(pd.DataFrame([row])[RISK_FEATURES])[0])
    score = max(0.0, min(1.0, score))
    return {"risk_score": round(score, 4), "risk_level": risk_level(score)}

@app.post("/api/priority")
def priority(req: PriorityRequest):
    row = {
        "Temp_C": req.temp_c,
        "Vibration_mm_s": req.vibration_mm_s,
        "Pressure_Bar": req.pressure_bar,
        "Acoustic_dB": req.acoustic_db,
        "Inspection_Duration_min": req.inspection_duration_min,
        "Downtime_Cost_USD": req.downtime_cost_usd,
        "Technician_Availability_pct": req.technician_availability_pct,
        "Risk_Score": req.risk_score,
        "Vibration_Deviation": req.vibration_mm_s - 2.5,
        "Vibration_High_Risk": int(req.vibration_mm_s > 4.0),
    }
    pred = int(PRIORITY_MODEL.predict(pd.DataFrame([row])[PRIORITY_FEATURES])[0])
    labels = {1: "Low", 2: "Medium", 3: "High"}
    windows = {1: "Within the next 2 weeks", 2: "Within the next 24-72 hours", 3: "Within the next 4 hours"}
    return {"priority": pred, "priority_label": labels[pred], "window": windows[pred]}

@app.post("/api/ask")
def ask(req: AskRequest):
    if req.failure_type and req.failure_type in KB:
        kb_row = KB[req.failure_type]
    else:
        matches = retrieve_kb(req.question, top_k=1)
        kb_row = matches[0] if matches else {}
    return {
        "answer": answer_kb_question(req.question, kb_row),
        "failure_type": kb_row.get("Failure_Type"),
        "kb": kb_row or None,
        "llm_used": has_openai(),
    }

@app.get("/api/knowledge/{failure_type}")
def knowledge(failure_type: str):
    row = KB.get(failure_type)
    if not row:
        raise HTTPException(status_code=404, detail=f"Unknown failure type: {failure_type}")
    return row

@app.get("/api/health")
def health():
    return {"status": "ok", "llm_available": has_openai(), "active_sessions": len(sessions)}

@app.post("/api/chat")
def chat(req: ChatRequest):
    session = get_session(req.session_id)

    if req.message.strip().lower() == "reset":
        sessions[req.session_id] = {f: None for f in CHAT_FIELDS}
        sessions[req.session_id]["_explained"] = False
        return {"reply": "Machine information cleared. Tell me about the machine whenever you're ready."}

    extracted = extract_sensor_readings(req.message)
    if extracted:
        for key, value in extracted.items():
            if value is not None:
                session[key] = value

    missing = [f for f in CHAT_FIELDS if session[f] is None]
    if missing:
        return {
            "reply": "I've recorded what you gave me so far. I still need:\n"
                      + "\n".join(f"- {FIELD_PROMPTS[m]}" for m in missing),
            "missing_fields": missing,
            "readings_so_far": session,
        }

    readings = {f: session[f] for f in CHAT_FIELDS}
    try:
        prediction = predict_failure(readings)
    except ValueError:
        session["Type"] = None
        return {
            "reply": "I didn't recognize that machine type. Please specify "
                     "L (low), M (medium), or H (high).",
            "missing_fields": ["Type"],
            "readings_so_far": session,
        }

    first_time = not session["_explained"]
    reply = explain_result(readings, prediction, question=req.message, first_time=first_time)
    session["_explained"] = True

    return {
        "reply": reply,
        "will_fail": prediction["will_fail"],
        "failure_type": prediction["failure_type"],
        "fail_prob": round(prediction["fail_prob"], 4),
        "readings_used": readings,
    }

@app.post("/api/chat/reset")
def chat_reset(session_id: str):
    sessions[session_id] = {f: None for f in CHAT_FIELDS}
    sessions[session_id]["_explained"] = False
    return {"status": "cleared", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)