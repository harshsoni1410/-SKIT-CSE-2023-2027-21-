"""
Week 4 - train the LipSense 3D CNN.

Harsh Soni, sprint task: "Model development - developed the initial AI model and
training pipeline."

Usage:
    # real data (once team_video_processing/dataset/ has samples)
    python -m team_ai_model.training.train --epochs 60 --batch 8

    # more augmentation for a very small dataset
    python -m team_ai_model.training.train --epochs 80 --augment-factor 5

    # pipeline smoke test, no dataset needed
    python -m team_ai_model.training.train --synthetic --epochs 2

Training augments ONLY the training split (flip / brightness / shift, x[1+factor]),
uses class weights for imbalance, ReduceLROnPlateau + EarlyStopping on val_accuracy.
A small self-collected dataset needs augmentation + many samples per word to work.

Outputs:
    team_ai_model/model/model_weights.h5     best model (by val_accuracy)
    team_ai_model/model/class_names.json     word order used for training
    team_ai_model/outputs/history.json       loss / accuracy per epoch
    team_ai_model/outputs/history.png        training curves
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "team_video_processing"))

# run either as a module (python -m team_ai_model.training.train) or as a script
if __package__:
    from .model import build_model, compile_model
    from .dataset import (
        load_dataset, make_synthetic, train_val_split, to_onehot, save_class_names,
    )
else:
    from team_ai_model.training.model import build_model, compile_model
    from team_ai_model.training.dataset import (
        load_dataset, make_synthetic, train_val_split, to_onehot, save_class_names,
    )

from augment import augment_batch  # noqa: E402  (team_video_processing/augment.py)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "team_ai_model" / "model"
OUTPUT_DIR = REPO_ROOT / "team_ai_model" / "outputs"
WEIGHTS_PATH = MODEL_DIR / "model_weights.h5"


def plot_history(history: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.plot(epochs, history["loss"], label="train")
    ax1.plot(epochs, history.get("val_loss", []), label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend(); ax1.grid(alpha=.3)
    ax2.plot(epochs, history["accuracy"], label="train")
    ax2.plot(epochs, history.get("val_accuracy", []), label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend(); ax2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the LipSense 3D CNN")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augment-factor", type=int, default=3,
                    help="augmented copies added per training sample (0 = off)")
    ap.add_argument("--synthetic", action="store_true",
                    help="train on generated noise (pipeline smoke test, no dataset)")
    args = ap.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        print("[synthetic] generating a fake dataset for a smoke test")
        X, y, class_names = make_synthetic(num_classes=3, per_class=8, seed=args.seed)
    else:
        print("[data] loading team_video_processing/dataset/")
        X, y, class_names = load_dataset()

    num_classes = len(class_names)
    print(f"classes ({num_classes}): {class_names}")
    print(f"samples: {len(X)}  input shape: {X.shape[1:]}")

    X_tr, y_tr, X_val, y_val = train_val_split(X, y, args.val_frac, args.seed)
    print(f"train: {len(X_tr)}   val: {len(X_val)}")

    # augment ONLY the training set (val stays real). Augmentation is what makes a
    # small self-collected dataset usable - flips / brightness / shifts multiply it.
    if args.augment_factor > 0 and not args.synthetic:
        before = len(X_tr)
        X_tr, y_tr = augment_batch(X_tr, y_tr, factor=args.augment_factor, seed=args.seed)
        print(f"augment: {before} -> {len(X_tr)} training samples (x{1 + args.augment_factor})")

    y_tr_oh = to_onehot(y_tr, num_classes)
    y_val_oh = to_onehot(y_val, num_classes)

    # class weights - if some words have fewer samples, weight them up
    counts = np.bincount(y_tr, minlength=num_classes)
    class_weight = {i: float(len(y_tr) / (num_classes * c)) if c else 1.0
                    for i, c in enumerate(counts)}

    save_class_names(class_names)

    import tensorflow as tf

    model = compile_model(build_model(num_classes), learning_rate=args.lr)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(WEIGHTS_PATH), monitor="val_accuracy", mode="max",
            save_best_only=True, save_weights_only=False, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=15, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=1,
        ),
    ]
    if len(X_val) == 0:
        callbacks = [tf.keras.callbacks.ModelCheckpoint(
            str(WEIGHTS_PATH), monitor="accuracy", mode="max",
            save_best_only=True, verbose=1)]
        val_data = None
    else:
        val_data = (X_val, y_val_oh)

    hist = model.fit(
        X_tr, y_tr_oh,
        validation_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )

    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    (OUTPUT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    try:
        plot_history(history, OUTPUT_DIR / "history.png")
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not plot history: {exc}")

    if not WEIGHTS_PATH.exists():
        model.save(WEIGHTS_PATH)

    best = max(history.get("val_accuracy", history["accuracy"]))
    print(f"\n[done] best {'val_' if val_data else ''}accuracy: {best:.3f}")
    print(f"  weights -> {WEIGHTS_PATH}")
    print(f"  history -> {OUTPUT_DIR / 'history.json'}")


if __name__ == "__main__":
    main()
