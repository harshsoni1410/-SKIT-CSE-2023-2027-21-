# LipSense — Build Roadmap (weekly)

Internal dev plan for building the codebase in weekly chunks. Each week lists what
gets added, in which module folder, so the work can be pushed to git one week at a
time. Code files carry a `# Week N` / `// Week N` marker comment at the top.

Module → owner (folder):
- `team_ai_model/` — Harsh Soni (AI model)
- `team_video_processing/` — Dhruv Sharma (video processing)
- `team_ui/` — Dipesh Yadav (UI: frontend + backend)
- `demo/` — whole team (integration)

The model input contract is fixed: **22 frames, 80×112, RGB, 0–1** —
`team_video_processing/preprocessing/preprocess.py` / [PRD.md §7](../PRD.md).

---

## Week 1 — done
- `team_ui/frontend/` — React + Vite + Tailwind scaffold, static layout, mock data.
- Components: Header, WebcamView, StatusBadge, PredictionCard, HistoryList, Controls, ErrorBanner.

## Week 2 — webcam + video pipeline functions
- **UI:** live webcam in `WebcamView` — `getUserMedia`, Start/Stop actually works,
  permission-denied → ErrorBanner, frame draw loop on a `<canvas>`.
- **Video:** `team_video_processing/face_detection/detector.py` — reusable
  `detect_face()` + `get_landmarks()` wrapping dlib (refactored out of `collect.py`).

## Week 3 — lip detection
- **UI:** MediaPipe FaceMesh in the browser → lip landmarks → lip ROI crop → 112×80,
  overlay box drawn on the canvas.
- **AI:** `team_ai_model/training/model.py` — `build_model()` 3D CNN
  (Conv3D→Pool3D→Conv3D→Pool3D→Flatten→Dense→Dropout→Softmax), per `docs/workflow.md`.

## Week 4 — speaking detection + training pipeline
- **UI:** speaking detection (inner-lip-distance ratio + calibrate button) and the
  22-frame sequence buffer; status idle→not_talking→recording→processing.
- **AI:** `team_ai_model/training/dataset.py` (loads `dataset/<word>/*.npy`, class
  order from folder names, train/val split) + `train.py` (compile, ModelCheckpoint on
  best val_accuracy, saves history). Smoke-tested on synthetic data.
- **Video:** `team_video_processing/augment.py` — flip, brightness jitter, small shift.

## Week 5 — backend + evaluation
- **Backend:** `team_ui/backend/` — FastAPI app: `GET /health`, `GET /vocab`,
  `WS /ws/predict`. Loads `model_weights.h5` + `class_names.json` if present, else a
  **stub predictor** so the whole chain runs before the real model exists.
- **AI:** `team_ai_model/training/evaluate.py` — confusion matrix + per-class accuracy
  → `team_ai_model/outputs/`. `predict.py` — `predict(tensor) → (word, confidence)`.

## Week 6 — frontend ↔ backend wiring
- **UI:** WebSocket client — send the 22-frame sequence, receive `{word, confidence}`,
  append to history, low-confidence (<0.6) → "Prediction uncertain".
- End-to-end test with the stub predictor. Remove `mockData.js`.

## Week 7 — standalone demo + dataset tools
- **Demo:** `demo/predict_live.py` — webcam → dlib → lip frames → preprocess → model →
  word on screen (no browser). `demo/validate_model.py` — accuracy + confusion matrix
  on known `.npy` samples.
- **Video:** `team_video_processing/data_collection/build_dataset.py` — turn raw
  recordings into the `.npy` dataset, write `class_names.json`.

## Week 8 — integration + polish
- Swap the stub for the real trained model, preprocessing parity check
  (browser crop vs `preprocess.py`).
- Error states, reconnect, mobile layout, loading states.
- README + `PROGRESS.md` + `docs/weekly/` updates, final demo recording notes.

---

## Push flow (each week)

```
git add <the week's files>
git commit -m "<type>(<module>): <what> [Week N]"
git pull --rebase origin main
git push origin main
```

Update `PROGRESS.md` and `docs/weekly/week-NN.md` in the same push.
