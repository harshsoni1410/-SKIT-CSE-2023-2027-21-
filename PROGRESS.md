# LipSense — Progress Log

Updated weekly. Newest entry on top.

---

## 2026-09-04 — Build sprint: frontend + model kickoff

Dhruv Sharma and Dipesh Yadav are on emergency family leave (~2–3 months). Harsh is
covering the video-processing and UI modules in the meantime; work is tracked against
[`docs/build-roadmap.md`](docs/build-roadmap.md).

**Done:**
- **UI (Week 1)** — `team_ui/frontend/` React + Vite + Tailwind scaffold; static layout
  with all components (Header, WebcamView, StatusBadge, PredictionCard, HistoryList,
  Controls, ErrorBanner) on mock data.
- **UI (Week 2)** — live webcam via `getUserMedia`, Start/Stop, permission/device error
  handling, canvas frame loop.
- **UI (Week 3)** — in-browser lip detection with MediaPipe FaceLandmarker
  (`src/lib/lipDetector.js`): lip ROI box overlay + 112×80 lip crop + inner-lip opening
  ratio, matching the `preprocess.py` contract.
- **Video (Week 2)** — `team_video_processing/face_detection/detector.py`: reusable
  `FaceLandmarkDetector` wrapping dlib face + 68-landmark detection (factored out of
  `collect.py`).
- **AI (Week 3)** — `team_ai_model/training/model.py`: 3D CNN (`build_model` /
  `compile_model`), 3× Conv3D→BN→Pool3D → GlobalAveragePooling3D → Dense → Dropout →
  Softmax. Self-test OK — output `(N, num_classes)`, softmax sums to 1, ~298 K params.
- **UI (Week 4)** — `Calibrate` button + speaking detection + 22-frame sequence buffer
  (`src/lib/speechCapture.js`, `src/hooks/useSpeechCapture.js`), mirroring `collect.py`'s
  calibrate → RECORDING → silence-ends-utterance logic. `sequenceToTensor()` builds the
  `(1,22,80,112,3)` float tensor. Status now idle→calibrating→not_talking→recording→processing.
- **AI (Week 4)** — `training/dataset.py` (loads `dataset/<word>/*.npy`, class order from
  folder names, stratified train/val split, `class_names.json`, synthetic generator) and
  `training/train.py` (ModelCheckpoint on best val_accuracy, EarlyStopping, history
  json + png). Smoke-tested: `train.py --synthetic --epochs 2` runs end to end, loss
  drops 2.07→0.87.
- **Video (Week 4)** — `team_video_processing/augment.py`: flip, brightness/contrast
  jitter, small spatial shift, `augment_batch()`. Self-test OK.
- **Env fix** — `venv/` was corrupted (TensorFlow + pip had missing files, likely
  OneDrive/AV). Recreated from Python 3.10 and reinstalled `requirements.txt`.

- **Backend (Week 5)** — `team_ui/backend/` FastAPI + WebSocket: `GET /health`,
  `GET /vocab`, `WS /ws/predict`. `predictor.py` loads the trained model if present,
  else a `StubPredictor` (random word) so the pipeline runs before the model exists.
  Tested: HTTP endpoints + WS (valid tensor → prediction, bad tensor → error).
- **AI (Week 5)** — `training/predict.py` (`LipReader` — model file → word + confidence
  + probs; `load_reader()` returns None when untrained) and `training/evaluate.py`
  (confusion matrix + per-class accuracy + `metrics.json` via sklearn). Eval pipeline
  smoke-tested with `--synthetic`.

- **UI (Week 6)** — `src/lib/predictClient.js` + `src/hooks/usePredictClient.js`: connects
  to the backend while the camera is on, sends the `sequenceToTensor` output on each
  utterance, shows `{word, confidence}` + a history entry, `<0.6` → "Prediction uncertain".
  `ConnectionBadge` shows backend state + stub/model. `mockData.js` removed; app starts
  empty. End-to-end tested against the stub backend (utterance → word).

- **Demo (Week 7)** — `demo/predict_live.py` (standalone webcam → 3D CNN → word on screen,
  reuses the shared detector + preprocess + `LipReader`), `demo/validate_model.py`
  (accuracy + per-class + text confusion matrix, `--min-accuracy` gate). Video:
  `data_collection/build_dataset.py` (recorded clips → `.npy` + `class_names.json`).
  Tested end to end with a fake dataset: build → train → validate → confusion matrix.
- **Polish (Week 8)** — WebSocket client auto-reconnects with backoff; "predicting…"
  state on the card; mobile layout tweaks (min-w-0, header wrap, smaller padding);
  `docs/preprocessing-parity.md` documents browser vs Python frame parity; backend
  swaps stub → trained model automatically when `model_weights.h5` appears (verified).
  Root `README.md` now has full run instructions.

**Status:** Weeks 1–8 of the build roadmap complete. Web demo + standalone demo + training
+ evaluation all run. Remaining real-world work: record an actual dataset, train, tune.

### Fixes after first real training run (random predictions)
- **Bug:** `train.py` never used `augment.py`. Now augments the training split
  (`--augment-factor`, default 3 → x4 data), plus class weights + `ReduceLROnPlateau`,
  higher default epochs (60) and lower LR (5e-4).
- Added `team_video_processing/inspect_sample.py` — saves a 22-frame montage of a `.npy`
  sample so you can check the recorded crops actually show a centred, moving mouth
  (bad data is the usual cause of random predictions).

**Next:**
- Record a balanced dataset (`collect.py` / `build_dataset.py`) — **50–100+ samples per
  word**, start with 4–5 visually different words. Check a few with `inspect_sample.py`.
  Then `train.py` → `validate_model.py` → demo with the real model.

**Blockers:** none. (Risk: OneDrive keeps corrupting `venv/` — plan to move the project
off OneDrive.)

---

## Week of 2026-09-03 — Foundation + first tasks

**Done:**
- Inspected the repo — no existing code, starting from scratch.
- Created folder structure: `team_ai_model/`, `team_video_processing/`, `team_ui/`, `demo/`, `docs/`.
- Python 3.10 virtual environment created (`venv/`), all dependencies installed
  (TensorFlow 2.10.1, OpenCV 4.8.1, NumPy 1.23.5, dlib 20.0.1).
- Downloaded dlib 68-point landmark model -> `team_ai_model/model/face_weights.dat`.
- `preprocessing/preprocess.py` self-test passes: tensor shape (1, 22, 80, 112, 3).
- **Lead first task** — `PRD.md` + `docs/workflow.md` (requirements + overall workflow).
- **Dhruv Sharma first task** — `team_video_processing/data_collection/collect.py` (collect video dataset).
- **Dipesh Yadav first task** — `team_ui/frontend/DESIGN.md` (basic UI design).
- Set up weekly Form-3 report tooling: `generate_report.py` + `.github/workflows/auto_weekly_report.yml`.
- Weekly logs now live in `docs/weekly/` — see [week-01.md](docs/weekly/week-01.md).

**Next sprint tasks:**
- Lead: dataset planning.
- Dhruv Sharma: video preprocessing.
- Dipesh Yadav: video upload interface (React + Vite + Tailwind scaffold).

**Blockers:** none.

---

## Task ownership — current "first tasks"

| Member | Sprint | First task | File |
|---|---|---|---|
| Harsh Soni (Lead) | AI Model Development | Project planning & requirements | `PRD.md`, `docs/workflow.md` |
| Dhruv Sharma | Video Processing & Data Preparation | Collect video dataset | `team_video_processing/data_collection/collect.py` |
| Dipesh Yadav | UI Integration & Development | Design user interface | `team_ui/frontend/DESIGN.md` |
