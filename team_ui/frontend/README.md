# LipSense Frontend

React + Vite + Tailwind CSS app for the live lip-reading demo.
UI design and rules: [DESIGN.md](DESIGN.md).

## Run

```
cd team_ui/frontend
npm install
npm run dev
```

Open http://localhost:5173. For live predictions also start the backend
(`cd team_ui/backend && uvicorn main:app --port 8000`) — without it the UI still
runs, it just shows "backend unreachable" when an utterance completes.

## Build

```
npm run build      # output in dist/
npm run preview     # serve the built app
```

## Structure

| Path | What |
|---|---|
| `src/App.jsx` | global state (camera, status, prediction, history, error) + layout |
| `src/components/` | `Header`, `WebcamView`, `StatusBadge`, `PredictionCard`, `HistoryList`, `Controls`, `ErrorBanner` |
| `src/lib/lipDetector.js` | MediaPipe FaceLandmarker wrapper — lip box + 112×80 crop + opening ratio |
| `src/lib/speechCapture.js` | speaking detection + 22-frame sequence buffer + `sequenceToTensor()` |
| `src/lib/predictClient.js` | WebSocket client for the backend `/ws/predict` |
| `src/hooks/` | `useSpeechCapture`, `usePredictClient` |
| `src/constants.js` | confidence threshold, WS URL (`VITE_WS_URL`), history limit |
| `tailwind.config.js` | dark theme tokens (`base`, `card`, `line`, `accent`) |

## Status (weekly)

- [x] **Week 1** — scaffold + static layout with mock data
- [x] **Week 2** — live webcam (`getUserMedia`), Start/Stop, permission errors, canvas frame loop
- [x] **Week 3** — in-browser lip detection (MediaPipe FaceLandmarker), lip ROI box + 112×80 crop
- [x] **Week 4** — calibration + speaking detection + 22-frame sequence buffer + `sequenceToTensor`
- [x] **Week 5** — backend built (`team_ui/backend/`, FastAPI + WS, stub predictor)
- [x] **Week 6** — WebSocket client wired: utterance → `/ws/predict` → word + confidence + history
- [x] **Week 7** — standalone `demo/predict_live.py` + `validate_model.py` + `build_dataset.py`
- [x] **Week 8** — auto-reconnect, "predicting…" state, mobile tweaks, preprocessing-parity doc

The frame sequence sent to the backend must match the contract in
[PRD.md § 7](../../PRD.md) / `team_video_processing/preprocessing/preprocess.py`:
22 frames, 80×112, RGB, values 0–1.
