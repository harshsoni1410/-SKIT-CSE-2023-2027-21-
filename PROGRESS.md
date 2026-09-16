# LipSense — Progress Log

Updated weekly. Newest entry on top.

---

## 2026-09-15 — Build 9: locking the real target (accurate similar-word detection)

(Note: "Build N" here is an internal dev-sprint counter, not a calendar week - it's
unrelated to the Form-3 weekly progress report's "Week N", which is based on real
calendar time since project start. This is Build 9, the actual current report is Week 3.)

Full build (Weeks 1-8) was already done; this week turns it toward the project's actual
point: correctly telling apart words that look almost identical on the lips.

**Decisions locked in:**
- **Vocabulary:** `bat, cat, hat, mat, rat, sat` — a rhyming minimal-pair set, chosen on
  purpose because it's the hard case (differ only in the first consonant), not a
  hedge-with-easy-words set. See `PRD.md` §4/§8 for the honest technical framing.
- **Dataset:** self-recorded via `collect.py`, no public dataset for now (fits the
  existing pipeline directly; can add transfer learning later if accuracy needs it).
- **Weekly reports:** generated locally on demand (`python generate_report.py weekly`)
  instead of relying on GitHub Actions, since work isn't being pushed yet (see below).

**Done:**
- `team_ai_model/training/model.py` — new `cnn_lstm` architecture (Conv3D → Bidirectional
  LSTM, keeps frame order instead of averaging it away) as the default, `cnn` (old
  design) kept for comparison via `--architecture`. Self-tested both, ~258K / ~297K
  params.
- `team_video_processing/augment.py` — added `time_warp` (speaking-speed jitter).
- `team_video_processing/data_collection/collect.py` — `--words w1,w2,...` session mode,
  automatic reject-on-bad-quality (dark frames / near-zero motion), `session_log.csv`.
- `team_ai_model/training/train.py` — `--architecture`, `--label-smoothing` flags;
  default `--augment-factor` 3 → 5.
- Full pipeline re-smoke-tested end to end (`train.py --synthetic --epochs 2`) with the
  new architecture — runs clean, ~13-19s/epoch on this CPU-only laptop for a tiny
  synthetic set (real training will take a lot longer; see "Next").
- `.github/workflows/auto_weekly_report.yml` — paused the Thursday schedule
  (`workflow_dispatch` still works for a manual run) since it runs against whatever is
  on GitHub, and nothing is being pushed there right now — left running it would just
  produce empty/stale reports on the remote.
- `PRD.md` / `docs/build-roadmap.md` updated with the vocabulary decision and the
  viseme-ambiguity limitation, written down before results come in rather than after.

**Recording protocol for the real dataset (the part only Harsh can do — needs the
physical webcam):**
- `python team_video_processing/data_collection/collect.py --words bat,cat,hat,mat,rat,sat --samples 25`
  run several times (different sittings) until each word has 100+ samples — don't do it
  all in one sitting, or the model will learn today's lighting/pose instead of the word.
- Say each word a little more deliberately than casual speech — don't full-on
  exaggerate, but a clear, unhurried mouth shape gives the model a bigger signal for the
  initial consonant, which is the entire thing it needs to tell these six words apart.
- Vary lighting / time of day / distance from camera across sittings on purpose.
- After each sitting, spot check a few new samples with
  `python team_video_processing/inspect_sample.py <path-to-a-.npy>` and check
  `dataset/<word>/session_log.csv` for a lot of "rejected" lines (means recalibrate `c`
  or fix lighting).

**Next:** once a real dataset exists, `train.py --epochs 80 --augment-factor 5` →
`evaluate.py` → `demo/validate_model.py`, then look at `confusion_matrix.png` — confusion
clustering inside viseme groups (bat/mat, cat/hat) is expected; confusion spread evenly
across all six means something upstream (data quality, calibration) needs fixing first.

**Blockers:** none. (Same OneDrive/`venv` corruption risk as before — still recommend
moving the project off OneDrive eventually.)

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
