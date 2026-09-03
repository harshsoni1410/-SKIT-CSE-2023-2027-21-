"""
Week 5 - prediction backend for LipSense.

Loads the trained 3D CNN if it exists; otherwise falls back to a STUB predictor that
returns a random word. The stub lets the whole browser -> WebSocket -> UI pipeline be
built and demoed before the AI model is ready (model lands later in the schedule).

`GET /health` reports which one is active.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# draft vocabulary (PRD.md section 4) - only used by the stub, before class_names.json exists
DRAFT_VOCAB = [
    "a", "bye", "can", "cat", "demo", "dog", "hello",
    "here", "is", "lips", "my", "read", "you",
]

INPUT_SHAPE = (22, 80, 112, 3)


class StubPredictor:
    trained = False
    kind = "stub"

    def __init__(self, vocab: list[str] | None = None):
        self.vocab = vocab or DRAFT_VOCAB

    def predict(self, tensor: np.ndarray) -> dict:
        word = random.choice(self.vocab)
        confidence = round(random.uniform(0.35, 0.95), 3)
        return {"word": word, "confidence": confidence, "stub": True}


class ModelPredictor:
    trained = True
    kind = "model"

    def __init__(self, reader):
        self.reader = reader
        self.vocab = list(reader.class_names)

    def predict(self, tensor: np.ndarray) -> dict:
        word, confidence, probs = self.reader.predict(tensor)
        return {"word": word, "confidence": round(confidence, 3), "stub": False, "probs": probs}


def load_predictor():
    """ModelPredictor if a trained model is on disk, else StubPredictor."""
    try:
        from team_ai_model.training.predict import load_reader

        reader = load_reader()
        if reader is not None:
            print(f"[predictor] loaded trained model - classes: {reader.class_names}")
            return ModelPredictor(reader)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[predictor] could not load model ({exc}); using stub")

    print("[predictor] no trained model - using STUB (random words)")
    return StubPredictor()


def coerce_tensor(data, shape=None) -> np.ndarray:
    """Flat list (+ optional shape) or nested list -> (1, 22, 80, 112, 3) float32."""
    arr = np.asarray(data, dtype="float32")
    if shape is not None:
        arr = arr.reshape(shape)
    if arr.ndim == len(INPUT_SHAPE):
        arr = arr[None, ...]
    if arr.shape[1:] != INPUT_SHAPE:
        raise ValueError(f"expected frames {INPUT_SHAPE}, got {arr.shape[1:]}")
    return arr
