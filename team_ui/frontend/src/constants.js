// Shared UI constants.

// Below this confidence the UI shows "Prediction uncertain" instead of the word
// (DESIGN.md "Confidence display rule", PRD.md FR-10).
export const CONFIDENCE_THRESHOLD = 0.6

// Backend WebSocket endpoint. Override with VITE_WS_URL in a .env file if needed.
export const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/predict'

// Backend HTTP base (for GET /vocab etc). Override with VITE_API_URL if needed.
export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// How many predictions to keep in the history list.
export const HISTORY_LIMIT = 10

// If the top two words are closer than this, the UI flags it as a "close call"
// (e.g. bat vs mat - visually similar on the lips).
export const CLOSE_CALL_MARGIN = 0.1
