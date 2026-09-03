"""
Week 5 - evaluate the trained LipSense model.

Harsh Soni, sprint task: evaluation - confusion matrix + per-class accuracy.

    python team_ai_model/training/evaluate.py                 # uses the real dataset + trained model
    python team_ai_model/training/evaluate.py --synthetic     # pipeline smoke test (untrained)

Writes to team_ai_model/outputs/:
    confusion_matrix.png
    per_class_accuracy.png
    metrics.json           accuracy, macro-F1, per-class precision/recall/accuracy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .model import build_model
    from .dataset import load_dataset, make_synthetic
    from .predict import LipReader
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from team_ai_model.training.model import build_model
    from team_ai_model.training.dataset import load_dataset, make_synthetic
    from team_ai_model.training.predict import LipReader

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "team_ai_model" / "outputs"


def _predict_all(predict_fn, X: np.ndarray, batch: int = 16) -> np.ndarray:
    preds = []
    for i in range(0, len(X), batch):
        preds.append(predict_fn(X[i:i + batch]))
    return np.concatenate(preds)


def plot_confusion(cm: np.ndarray, classes: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.1 * len(classes) + 2, 1.0 * len(classes) + 2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix")
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class_accuracy(acc: dict[str, float], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(acc.keys())
    vals = [acc[n] for n in names]
    fig, ax = plt.subplots(figsize=(1.0 * len(names) + 2, 3.6))
    ax.bar(names, vals, color="#4E79A7")
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("Per-class accuracy")
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the LipSense model")
    ap.add_argument("--synthetic", action="store_true",
                    help="use generated data + an untrained model (smoke test)")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from sklearn.metrics import (
        confusion_matrix, classification_report, accuracy_score, f1_score,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        print("[synthetic] generated data + untrained model - checking the eval pipeline only")
        X, y, classes = make_synthetic(num_classes=4, per_class=6, seed=1)
        model = build_model(len(classes))
        predict_fn = lambda xb: model.predict(xb, verbose=0).argmax(axis=1)  # noqa: E731
    else:
        X, y, classes = load_dataset()
        reader = LipReader()
        if reader.class_names != classes:
            print(f"[warn] class order mismatch:\n  dataset={classes}\n  model={reader.class_names}")
            classes = reader.class_names
        predict_fn = lambda xb: reader.model.predict(xb, verbose=0).argmax(axis=1)  # noqa: E731

    y_pred = _predict_all(predict_fn, X, args.batch)

    labels = list(range(len(classes)))
    cm = confusion_matrix(y, y_pred, labels=labels)
    overall = float(accuracy_score(y, y_pred))
    macro_f1 = float(f1_score(y, y_pred, labels=labels, average="macro", zero_division=0))
    report = classification_report(
        y, y_pred, labels=labels, target_names=classes, output_dict=True, zero_division=0,
    )

    per_class_acc = {}
    for i, cls in enumerate(classes):
        total = cm[i].sum()
        per_class_acc[cls] = float(cm[i, i] / total) if total else 0.0

    plot_confusion(cm, classes, OUTPUT_DIR / "confusion_matrix.png")
    plot_per_class_accuracy(per_class_acc, OUTPUT_DIR / "per_class_accuracy.png")

    metrics = {
        "overall_accuracy": overall,
        "macro_f1": macro_f1,
        "classes": classes,
        "n_samples": int(len(X)),
        "per_class_accuracy": per_class_acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\noverall accuracy: {overall:.3f}   macro-F1: {macro_f1:.3f}")
    for cls, a in per_class_acc.items():
        print(f"  {cls:10s} {a:.3f}")
    print(f"\nsaved -> {OUTPUT_DIR}/confusion_matrix.png, per_class_accuracy.png, metrics.json")


if __name__ == "__main__":
    main()
