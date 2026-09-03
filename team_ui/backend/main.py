"""
Week 5 - LipSense backend (FastAPI + WebSocket).

Run from this folder:

    cd team_ui/backend
    uvicorn main:app --reload --port 8000

Routes (team_ui/README.md):
    GET  /health        -> { status, model: "model" | "stub", vocab_size }
    GET  /vocab         -> { vocab: [...], count }
    WS   /ws/predict    -> send { data: [...floats], shape: [1,22,80,112,3] }
                           recv { word, confidence, stub }

The frontend sends only a compact 22-frame lip sequence when an utterance ends,
never raw webcam video.
"""

from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from predictor import load_predictor, coerce_tensor, INPUT_SHAPE

app = FastAPI(title="LipSense API", version="0.1.0")

# dev: the Vite frontend runs on a different port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = load_predictor()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": predictor.kind,          # "model" or "stub"
        "trained": predictor.trained,
        "vocab_size": len(predictor.vocab),
        "input_shape": list(INPUT_SHAPE),
    }


@app.get("/vocab")
def vocab():
    return {"vocab": predictor.vocab, "count": len(predictor.vocab)}


@app.websocket("/ws/predict")
async def ws_predict(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "ready", "model": predictor.kind})
    try:
        while True:
            msg = await ws.receive_json()
            data = msg.get("data")
            shape = msg.get("shape")
            if data is None:
                await ws.send_json({"error": "missing 'data'"})
                continue
            try:
                tensor = coerce_tensor(data, shape)
            except Exception as exc:
                await ws.send_json({"error": f"bad tensor: {exc}"})
                continue

            result = predictor.predict(tensor)
            await ws.send_json({"type": "prediction", **result})
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
