# LipSense Frontend — UI Design

> Member 2, first sprint task: "Design user interface — design the basic interface for the
> application." This is the design/wireframe. The React + Vite + Tailwind implementation comes
> in later tasks.

## Goal

A single-page web app for a live demo: open the app, allow the webcam, silently speak a word,
see the predicted word + confidence, and keep a short history.

## Tech

- React (functional components + hooks)
- Vite (dev server + build)
- Tailwind CSS (styling)
- Webcam via the browser `navigator.mediaDevices.getUserMedia`
- Talks to the FastAPI backend over WebSocket (later task)

## Layout (desktop)

```
+--------------------------------------------------------------------------+
|  LipSense                              Real-Time AI Lip Reading (3D CNN)  |   <- Header
+--------------------------------------------------------------------------+
|                                        |                                 |
|                                        |   STATUS                        |
|         +------------------------+     |   [ NOT TALKING ]               |   <- status badge
|         |                        |     |                                 |
|         |     Webcam preview     |     |   PREDICTION                    |
|         |   (lip ROI box drawn)  |     |        HELLO                    |   <- big word
|         |                        |     |   Confidence  ####----  78%    |   <- bar
|         +------------------------+     |                                 |
|                                        |   HISTORY                       |
|   [ Start Camera ] [ Stop ] [ Clear ]  |   hello    92%   12:04:01       |   <- list
|                                        |   dog      85%   12:03:44       |
|   error / permission messages here     |   cat      61%   12:03:20       |
|                                        |                                 |
+--------------------------------------------------------------------------+
```

On mobile the two columns stack vertically (webcam on top, panel below).

## Components

| Component | Responsibility |
|---|---|
| `App` | holds global state (camera on/off, current prediction, history) |
| `Header` | title + tagline |
| `WebcamView` | `<video>` element, requests camera, draws the lip ROI overlay on a `<canvas>` |
| `StatusBadge` | shows `NOT TALKING` / `RECORDING` / `PROCESSING` with a color |
| `PredictionCard` | large predicted word + confidence bar; shows "Prediction uncertain" if confidence < 0.6 |
| `HistoryList` | last ~10 predictions (word, confidence, time) |
| `Controls` | Start Camera / Stop Camera / Clear History buttons |
| `ErrorBanner` | camera permission denied, backend disconnected, etc. |

## State (in `App`)

```
cameraOn: boolean
status: "idle" | "not_talking" | "recording" | "processing"
prediction: { word: string, confidence: number } | null
history: Array<{ word: string, confidence: number, time: string }>
error: string | null
```

## Visual style

- Dark theme: background near-black (`#0b0f14`), cards `#151b23`.
- One accent color (teal / indigo) for the confidence bar and active buttons.
- Large, readable font for the predicted word (e.g. `text-5xl font-bold`).
- Rounded cards, subtle border, small shadow.
- Keep it minimal — this is a viva demo, not a product.

## Confidence display rule

- `confidence >= 0.6` -> show the word normally.
- `confidence < 0.6` -> show "Prediction uncertain" in a muted color, still list the top guess small.

## Out of scope for this task

- Actual React code / project scaffold (next task).
- WebSocket wiring to the backend (task 6).
- Authentication, database.
