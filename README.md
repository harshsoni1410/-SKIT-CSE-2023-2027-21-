# LipSense — Real-Time AI Lip Reading System Using 3D CNN

Final-year B.Tech (CSE) project — SKIT, 2023–2027, Group 21.

LipSense recognises a **spoken word** from a person's **lip movement** on a webcam, without
using audio. It is a **word-level** system (not full-sentence transcription).

## Pipeline

```
Webcam -> Face detection -> Facial landmarks -> Lip ROI crop ->
Speaking detection -> Frame sequence -> Preprocess -> 3D CNN -> Word + Confidence -> UI
```

Details: [docs/workflow.md](docs/workflow.md)

## Team & folders

| Folder | Owner | Sprint |
|---|---|---|
| [team_ai_model/](team_ai_model/) | Harsh Soni (Team Lead) | AI Model Development |
| [team_video_processing/](team_video_processing/) | Dhruv Sharma | Video Processing & Data Preparation |
| [team_ui/](team_ui/) | Dipesh Yadav | UI Integration & Development |
| [demo/](demo/) | Whole team | Real-time prediction (integration) |

Plan: [docs/team_plan.md](docs/team_plan.md) · Requirements: [PRD.md](PRD.md) · Progress: [PROGRESS.md](PROGRESS.md) · Weekly: [docs/weekly/](docs/weekly/)

## Tech stack

- **AI/ML:** Python 3.10, TensorFlow/Keras, 3D CNN, NumPy, scikit-learn
- **Computer Vision:** OpenCV, dlib (68-point landmarks), MediaPipe FaceLandmarker (browser)
- **Backend:** FastAPI + WebSocket
- **Frontend:** React + Vite + Tailwind CSS

## Run

```
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt

# 1. web demo — backend + frontend (two terminals)
cd team_ui/backend  && uvicorn main:app --port 8000
cd team_ui/frontend && npm install && npm run dev        # http://localhost:5173

# 2. standalone webcam demo (no browser)
python demo/predict_live.py

# 3. train / evaluate (needs a dataset in team_video_processing/dataset/)
python -m team_ai_model.training.train --epochs 40
python demo/validate_model.py
```

Before a model is trained the backend serves a **stub predictor** (random word) so the
full pipeline still runs. Build progress: [PROGRESS.md](PROGRESS.md),
[docs/build-roadmap.md](docs/build-roadmap.md).
