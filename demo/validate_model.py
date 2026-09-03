"""
Week 7 - validate the trained model on the known .npy dataset.

Whole team - must pass before wiring the frontend to a real model.

    python demo/validate_model.py
    python demo/validate_model.py --min-accuracy 0.6

Prints overall accuracy, per-class accuracy and a text confusion matrix.
Exits 1 if accuracy is below --min-accuracy (so it can gate CI later).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from team_ai_model.training.dataset import load_dataset  # noqa: E402
from team_ai_model.training.predict import load_reader  # noqa: E402


def text_confusion(cm: np.ndarray, classes: list[str]) -> str:
    w = max(len(c) for c in classes + ["true\\pred"])
    head = "true\\pred".ljust(w) + " | " + " ".join(c.rjust(4) for c in classes)
    rows = [head, "-" * len(head)]
    for i, c in enumerate(classes):
        rows.append(c.ljust(w) + " | " + " ".join(str(int(v)).rjust(4) for v in cm[i]))
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the LipSense model")
    ap.add_argument("--min-accuracy", type=float, default=0.0,
                    help="exit 1 if overall accuracy is below this")
    args = ap.parse_args()

    reader = load_reader()
    if reader is None:
        sys.exit("No trained model. Run: python -m team_ai_model.training.train --epochs 40")

    X, y, classes = load_dataset()
    if reader.class_names != classes:
        print(f"[warn] class order differs:\n  dataset={classes}\n  model={reader.class_names}")
        classes = reader.class_names

    probs = reader.model.predict(X, verbose=0)
    y_pred = probs.argmax(axis=1)

    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y, y_pred):
        cm[t, p] += 1

    overall = float((y_pred == y).mean())
    print(f"\nsamples: {len(X)}   classes: {classes}")
    print(f"overall accuracy: {overall:.3f}\n")
    for i, c in enumerate(classes):
        tot = cm[i].sum()
        acc = cm[i, i] / tot if tot else 0.0
        print(f"  {c:12s} {acc:.3f}  ({cm[i, i]}/{tot})")
    print("\nconfusion matrix:\n" + text_confusion(cm, classes))

    if overall < args.min_accuracy:
        sys.exit(f"\nFAIL: accuracy {overall:.3f} < required {args.min_accuracy:.3f}")
    print("\nOK")


if __name__ == "__main__":
    main()
