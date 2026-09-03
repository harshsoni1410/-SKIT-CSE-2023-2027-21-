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
- **Computer Vision:** OpenCV, dlib (68-point facial landmarks)
- **Backend:** FastAPI + WebSocket
- **Frontend:** React + Vite + Tailwind CSS
