"""
Week 5 - load a trained LipSense model and turn a lip-frame sequence into a word.

Harsh Soni, sprint task: system integration (inference).

    from team_ai_model.training.predict import load_reader
    reader = load_reader()                 # None if no trained model yet
    word, confidence, probs = reader.predict(tensor)   # tensor (1,22,80,112,3) or (22,...)

The demo and the backend both use this - one code path from model file to word.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import INPUT_SHAPE

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "team_ai_model" / "model"
WEIGHTS_PATH = MODEL_DIR / "model_weights.h5"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"


class LipReader:
    """Holds the loaded Keras model + the word list."""

    def __init__(self, weights_path: str | Path = WEIGHTS_PATH,
                 class_names_path: str | Path = CLASS_NAMES_PATH):
        import tensorflow as tf

        weights_path = Path(weights_path)
        class_names_path = Path(class_names_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"No trained model at {weights_path}")
        if not class_names_path.exists():
            raise FileNotFoundError(f"No class_names.json at {class_names_path}")

        self.model = tf.keras.models.load_model(weights_path)
        self.class_names: list[str] = json.loads(class_names_path.read_text(encoding="utf-8"))

    def predict(self, tensor) -> tuple[str, float, list[float]]:
        """
        tensor -> (word, confidence, full_probability_vector)

        Accepts (22, 80, 112, 3) or (1, 22, 80, 112, 3), float32, values 0..1.
        """
        x = np.asarray(tensor, dtype="float32")
        if x.ndim == len(INPUT_SHAPE):
            x = x[None, ...]
        if x.shape[1:] != INPUT_SHAPE:
            raise ValueError(f"expected frame shape {INPUT_SHAPE}, got {x.shape[1:]}")

        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        return self.class_names[idx], float(probs[idx]), [float(p) for p in probs]


def load_reader(weights_path: str | Path = WEIGHTS_PATH,
                class_names_path: str | Path = CLASS_NAMES_PATH) -> LipReader | None:
    """Return a LipReader, or None if the model has not been trained yet."""
    try:
        return LipReader(weights_path, class_names_path)
    except FileNotFoundError as exc:
        print(f"[predict] {exc} - no model available yet")
        return None


if __name__ == "__main__":
    reader = load_reader()
    if reader is None:
        print("Train a model first:  python -m team_ai_model.training.train --epochs 40")
    else:
        print("classes:", reader.class_names)
        dummy = np.zeros(INPUT_SHAPE, dtype="float32")
        word, conf, _ = reader.predict(dummy)
        print(f"dummy prediction: {word}  ({conf:.2f})")
