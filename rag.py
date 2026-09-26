import os
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent
KB_PATH = ROOT / "data/failure_knowledge_base.csv"
BINARY_MODEL_PATH = ROOT / "Model/binary_failure_model.pkl"
MULTICLASS_MODEL_PATH = ROOT / "Model/multiclass_failure_model.pkl"

Features = ["Type", "Air temperature K", "Process temperature K", "Rotational speed rpm", "Torque Nm", "Tool wear min"]
type_map = {'L': 0, 'M': 1, 'H': 2}

def load_pipeline():
    for path in (KB_PATH, BINARY_MODEL_PATH, MULTICLASS_MODEL_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError()


    kb = pd.read_csv(KB_PATH)
    binary_model = joblib.load(BINARY_MODEL_PATH)
    multiclass_model = joblib.load(MULTICLASS_MODEL_PATH)
    retrieve = _build_retriever(kb)
    return {
        'kb' : kb,
        "binary_model": binary_model["model"],
        "binary_threshold": binary_model.get("threshold", 0.5),
        "multiclass_model": multiclass_model["model"],
        "label_encoder": multiclass_model["label_encoder"],
        "retrieve": retrieve,
    }

def _build_retriever(kb):
    def row_to_text(row):
        return(
            f"Failure type {row['Failure_Type']}. Root cause: {row['Root_Cause']}. "
            f"Priority: {row['Priority']}. Recommendations: {row['Recommendation_1']}; "
            f"{row['Recommendation_2']}; {row['Recommendation_3']}."
        )

    docs = kb.apply(row_to_text, axis= 1).tolist()
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(docs)

    def retrieve(query, top_k=1):
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        results = kb.iloc[top_idx].copy()
        results['similarity'] = sims[top_idx]
        return results
    return retrieve


def openai_key():
    return bool(os.getenv('OPENAI_API_KEY'))

def _call_openai(prompt, model='gpt-4o-mini'):
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
         max_tokens=300,
        )
    return response.choices[0].message.content


def predict_and_recommand(sensor_input: dict, pipeline: dict, use_llm: bool = False):
    row = sensor_input.copy()
    row['Type'] = type_map.get(row['Type'], row['Type'])
    X =  pd.DataFrame([row])[Features]

    fail_prob = pipeline['binary_model'].predict_proba(X)[0, 1]
    will_fail = bool(fail_prob >= pipeline['binary_threshold'])

    if not will_fail:
        return{
            'will_fail': False,
            'failure_type': None,
            'answer': f'No failure predicted (failure probability: {fail_prob:.2f}).'
        }

    pred_idx = pipeline['multiclass_model'].predict(X)[0]
    failure_type = pipeline['label_encoder'].inverse_transform([pred_idx])[0]
    match = pipeline['retrieve'](f'failure type {failure_type}', top_k= 1).iloc[0]

    if use_llm and openai_key():
        prompt = (
            "You are a factory maintenance assistant. Using ONLY the context below, "
            "write a short (3-4 sentence) recommendation for a technician.\n\n"
            f"Sensor reading: {sensor_input}\n"
            f"Predicted failure type: {failure_type}\n"
            f"Knowledge base context: {json.dumps(match.to_dict(), default=str)}"
        )
        answer = _call_openai(prompt)
    else:
        recs = [match['Recommendation_1'], match['Recommendation_2'], match['Recommendation_3']]
        answer = (
            f'Predicted failure type: {failure_type} (probability: {fail_prob:.2f})\n'
            f"Root cause: {match['Root_Cause']}\n"
            f"Priority: {match['Priority']}\n"
            f"Estimated cost: ${match['Estimated_Cost_USD']:,.0f} ({match['Cost_Percentage']}% of total)\n"
            'Recommended actions:\n' + '\n'.join(f' - {r}' for r in recs)
        )
    return{'will_fail': True, 'failure_type': failure_type, 'answer': answer}


def answer_qustion(question: str, pipeline: dict, use_llm: bool = False):
    match = pipeline['retrieve'](question, top_k=1).iloc[0]

    if use_llm and openai_key():
        prompt = (
            "You are a factory maintenance assistant. Answer the technician's question "
            "using ONLY the context below. If the context doesn't really answer it, say so.\n\n"
            f"Question: {question}\n"
            f"Knowledge base context: {json.dumps(match.to_dict(), default=str)}"
        )
        return _call_openai(prompt)

    return(
        f"Closest match: {match['Failure_Type']} (similarity: {match['similarity']:.2f})\n"
        f"Root cause: {match['Root_Cause']}\n"
        f"Priority: {match['Priority']}\n"
        f"Recommendations: {match['Recommendation_1']}; {match['Recommendation_2']}; "
        f"{match['Recommendation_3']}"
        )

if __name__ == '__main__':
    pipeline = load_pipeline()
    # demo = {"Type": "L", "Air temperature K": 301.5, "Process temperature K": 310.9,
    #         "Rotational speed rpm": 2760, "Torque Nm": 8.0, "Tool wear min": 15}
    # result = predict_and_recommand(demo, pipeline, use_llm= True)
    # print(result["answer"])
    # print("Type 'q' to quit.\n")
    while True:
        question = input("You: ")
        if question.lower() == 'q':
            break

        answer = answer_qustion(question, pipeline, use_llm=True)
        print(answer)
