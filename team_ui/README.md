# team_ui — UI Integration & Development

**Owner:** Dipesh Yadav · **Sprint:** UI Integration & Development

| Path | Contents |
|---|---|
| `frontend/` | React + Vite + Tailwind CSS app (`DESIGN.md` holds the current UI design) |
| `backend/`  | FastAPI + WebSocket server for model inference |

## Frontend — components

1. Header — **LipSense** title + tagline.
2. Webcam preview (`<video>` + `getUserMedia`).
3. Face/lip ROI overlay (canvas) — optional.
4. Speaking status badge — `NOT TALKING` / `RECORDING` / `PROCESSING`.
5. Predicted word (large) + confidence % bar.
6. Prediction history list (word, confidence, time).
7. Controls — Start camera, Stop camera, Clear history.
8. Error / permission-denied messages.

Simple, modern, final-year demo style. Dark theme + one accent color.

## Backend — endpoints

| Route | Purpose |
|---|---|
| `GET /health` | server up check |
| `GET /vocab` | return the class list |
| `WS /ws/predict` | receive a lip-frame sequence -> 3D CNN -> `{word, confidence}` |

Load the model once at startup (`team_ai_model/model/model_weights.h5`).
Do not send raw webcam video to the backend — send only a compact lip-frame sequence when an
utterance completes.

## Responsibilities (sprint order)

1. Basic UI design + layout — `frontend/DESIGN.md`. *(current)*
2. Video upload interface.
3. Video preview/playback.
4. Prediction + confidence display.
5. Result history.
6. Backend + model integration (WebSocket).
7. Testing & polish.

## Setup

Frontend is scaffolded (React + Vite + Tailwind, needs Node.js LTS):

```
cd team_ui/frontend
npm install
npm run dev
```

See [frontend/README.md](frontend/README.md) for the weekly build status.
Backend (FastAPI + WebSocket) is added around Week 5.
