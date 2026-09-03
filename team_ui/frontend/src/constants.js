// Shared UI constants.

// Below this confidence the UI shows "Prediction uncertain" instead of the word
// (DESIGN.md "Confidence display rule", PRD.md FR-10).
export const CONFIDENCE_THRESHOLD = 0.6

// Backend WebSocket endpoint. Override with VITE_WS_URL in a .env file if needed.
export const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/predict'

// How many predictions to keep in the history list.
export const HISTORY_LIMIT = 10
