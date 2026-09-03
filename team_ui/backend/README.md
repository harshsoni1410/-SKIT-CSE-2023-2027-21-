# LipSense Backend

FastAPI + WebSocket server. Wraps the AI model; falls back to a random **stub**
predictor until the model is trained, so the frontend can be built end to end now.

## Run

```
cd team_ui/backend
uvicorn main:app --reload --port 8000
```

(dependencies are in the repo root `requirements.txt` — `fastapi`, `uvicorn`,
`websockets`; use the project `venv`.)

## Routes

| Route | Purpose |
|---|---|
| `GET /health` | `{ status, model: "model"｜"stub", trained, vocab_size, input_shape }` |
| `GET /vocab` | `{ vocab: [...words], count }` |
| `WS /ws/predict` | send `{ data: [floats], shape: [1,22,80,112,3] }` → recv `{ type:"prediction", word, confidence, stub }` |

## Model vs stub

`predictor.load_predictor()` checks for `team_ai_model/model/model_weights.h5` +
`class_names.json`:
- present → `ModelPredictor` (real 3D CNN via `team_ai_model/training/predict.py`)
- missing → `StubPredictor` (random word from the draft vocab)

`GET /health` shows which is active.

## Files

| File | What |
|---|---|
| `main.py` | FastAPI app + routes |
| `predictor.py` | model / stub loading, tensor validation |

## Status (weekly)

- [x] **Week 5** — FastAPI app, `/health`, `/vocab`, `WS /ws/predict`, stub fallback
- [x] **Week 6** — frontend WebSocket client wired; end-to-end tested with the stub
- [x] **Week 8** — model auto-loads when trained (no code change); client auto-reconnects
