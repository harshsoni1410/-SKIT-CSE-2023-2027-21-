# LipSense Frontend

React + Vite + Tailwind CSS app for the live lip-reading demo.
UI design and rules: [DESIGN.md](DESIGN.md).

## Run

```
cd team_ui/frontend
npm install
npm run dev
```

Open http://localhost:5173.

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
| `src/hooks/useSpeechCapture.js` | React wrapper for `speechCapture` |
| `src/mockData.js` | temporary sample data — removed once webcam + backend are wired in |
| `tailwind.config.js` | dark theme tokens (`base`, `card`, `line`, `accent`) |

## Status (weekly)

- [x] **Week 1** — scaffold + static layout with mock data
- [x] **Week 2** — live webcam (`getUserMedia`), Start/Stop, permission errors, canvas frame loop
- [x] **Week 3** — in-browser lip detection (MediaPipe FaceLandmarker), lip ROI box + 112×80 crop
- [x] **Week 4** — calibration + speaking detection + 22-frame sequence buffer + `sequenceToTensor`
- [ ] **Week 5** — connect to FastAPI backend over WebSocket (stub predictor)
- [ ] **Week 6** — real model, preprocessing parity check
- [ ] **Week 7** — testing, error states, mobile, polish

The frame sequence sent to the backend must match the contract in
[PRD.md § 7](../../PRD.md) / `team_video_processing/preprocessing/preprocess.py`:
22 frames, 80×112, RGB, values 0–1.
