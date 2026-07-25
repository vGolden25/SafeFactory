# SafeFactory

A predictive-maintenance API for industrial machines. Combines trained ML models (failure prediction, risk scoring, maintenance priority) with a conversational assistant that explains results in plain, non-technical language — grounded in a maintenance knowledge base.

## What it does

- **Predicts machine failure** from sensor readings (air/process temperature, rotational speed, torque, tool wear) using a two-stage model: a binary "will it fail?" classifier, followed by a multiclass model that identifies *which* failure type if so.
- **Scores equipment risk** and **maintenance priority** from separate trained models for broader asset monitoring.
- **Chats naturally**: extracts sensor readings from free-text messages via an LLM, accumulates them across turns per session, and explains predictions using a retrieval-augmented knowledge base (TF-IDF search over failure types, causes, costs, and recommendations).

## Project structure

```
SafeFactory/
├── python_server/
│   ├── main.py           # FastAPI app — all endpoints live here
│   └── .env              # OPENAI_API_KEY (not committed)
├── data/
│   └── failure_knowledge_base.csv   # KB used for retrieval
├── Model/
│   ├── binary_failure_model.pkl
│   ├── multiclass_failure_model.pkl
│   ├── risk_score_model.pkl
│   └── maintenance_priority_model.pkl
└── NoteBooks/            # training / EDA notebooks
```

## Setup

1. Install dependencies:
   ```bash
   pip install fastapi uvicorn pandas scikit-learn joblib python-dotenv openai
   ```

2. Create a `.env` file inside `python_server/`:
   ```
   OPENAI_API_KEY=your_key_here
   ```
   (Optional — the app still runs without it, falling back to non-LLM plain-text answers.)

3. Run the server:
   ```bash
   cd python_server
   python main.py
   ```
   or
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. Open the interactive API docs:
   ```
   http://127.0.0.1:8000/docs
   ```

## API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/predict` | POST | Predict failure from raw sensor values |
| `/api/risk` | POST | Compute an equipment risk score |
| `/api/priority` | POST | Compute maintenance priority level |
| `/api/ask` | POST | Ask a knowledge-base question about a failure type |
| `/api/chat` | POST | Multi-turn conversational assistant (extracts readings, predicts, explains) |
| `/api/chat/reset` | POST | Clear a chat session's stored readings |
| `/api/knowledge/{failure_type}` | GET | Look up a knowledge-base entry directly |
| `/api/health` | GET | Server status, LLM availability, active session count |

### Example: `/api/chat`

```json
{
  "session_id": "user1",
  "message": "Machine type L, air temp 301.5, process temp 310.9, speed 2760 rpm, torque 8 Nm, tool wear 15 min"
}
```

Returns a plain-language explanation of whether the machine is healthy or at risk of a specific failure type, along with the prediction confidence and recommended actions.

## Notes

- Sensor extraction and explanations use OpenAI's API when a key is available; without one, the app falls back to structured, non-LLM responses using the knowledge base directly.
- Chat session state is stored in memory and resets when the server restarts.
