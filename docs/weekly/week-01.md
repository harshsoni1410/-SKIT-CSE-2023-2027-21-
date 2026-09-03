# Week 1 — 2026-09-03 → 2026-09-09

**Theme:** Project foundation + each member's first sprint task.

## Planned

| Member | Sprint task | Deliverable |
|---|---|---|
| Harsh Soni (Lead) | Project planning & requirements | `PRD.md`, `docs/workflow.md` |
| Member 1 | Collect video dataset | `team_video_processing/data_collection/collect.py` |
| Member 2 | Design user interface | `team_ui/frontend/DESIGN.md` |

## Done

### Harsh Soni (Lead) — planning & requirements
- Wrote `PRD.md`: problem statement, scope, functional requirements, and the
  **model input contract** (frame count, size, channels, normalization) that all
  three modules must follow.
- Wrote `docs/workflow.md`: full webcam → 3D CNN → UI pipeline with viva Q&A.
- Set up repo structure, `requirements.txt` (Python 3.10), `.gitignore`.
- Set up the weekly Form-3 report tooling (`generate_report.py` + GitHub Actions).

### Member 1 — data collection tool
- Wrote `team_video_processing/data_collection/collect.py`: webcam capture,
  dlib face + 68-landmark detection, lip ROI crop, automatic utterance detection
  (calibrated inner-lip-distance threshold), saves each sample as
  `dataset/<word>/NNN.npy`.
- Wrote the shared `team_video_processing/preprocessing/preprocess.py` used by
  collection, training and the live demo (one preprocessing path everywhere).

### Member 2 — UI design
- Wrote `team_ui/frontend/DESIGN.md`: layout wireframe, component list, app state
  shape, visual style, and the low-confidence display rule.

## Environment set up this week
- Python 3.10 venv, all dependencies installed
  (TensorFlow 2.10.1, OpenCV 4.8.1, NumPy 1.23.5, dlib 20.0.1, reportlab 5.0.1).
- dlib 68-point landmark model downloaded → `team_ai_model/model/face_weights.dat`.
- `preprocess.py` self-test passes → tensor shape `(1, 22, 80, 112, 3)`.

## Next week (Week 2)
- Member 1: video preprocessing — turn raw recordings into clean input frames, test `collect.py` end to end and record a first small batch.
- Lead: dataset planning — vocabulary lock, samples-per-word target, augmentation plan.
- Member 2: scaffold the React + Vite + Tailwind project and build the static layout.
