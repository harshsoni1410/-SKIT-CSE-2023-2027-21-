"""
Week 4 - train the LipSense 3D CNN.

Harsh Soni, sprint task: "Model development - developed the initial AI model and
training pipeline."

Usage:
    # real data (once team_video_processing/dataset/ has samples)
    python -m team_ai_model.training.train --epochs 60 --batch 8

    # more augmentation for a very small dataset
    python -m team_ai_model.training.train --epochs 80 --augment-factor 8

    # compare against the original order-agnostic architecture
    python -m team_ai_model.training.train --architecture cnn

    # pipeline smoke test, no dataset needed
    python -m team_ai_model.training.train --synthetic --epochs 2

    # a run got cut off (laptop slept / shut down)? continue it up to --epochs total
    python -m team_ai_model.training.train --epochs 40 --augment-factor 2 --resume

Training augments ONLY the training split (flip / time-warp / brightness / contrast /
shift, x[1+factor]), uses class weights for imbalance, ReduceLROnPlateau + EarlyStopping
on val_accuracy. Default architecture is cnn_lstm (BiLSTM over frame order - see
model.py) since the target vocabulary is now visually-similar minimal pairs
(cat/bat/hat/mat/rat/sat) that only differ in the first couple of frames.

Outputs:
    team_ai_model/model/model_weights.h5     best model (by val_accuracy)
    team_ai_model/model/class_names.json     word order used for training
    team_ai_model/outputs/history.json       loss / accuracy per epoch
    team_ai_model/outputs/history.png        training curves
    team_ai_model/outputs/training_log.csv   per-epoch log, written live during training
"""

from __future__ import annotations

import argparse
import csv
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
TRAINING_LOG = OUTPUT_DIR / "training_log.csv"


def read_training_log() -> list[dict]:
    """Rows of training_log.csv (one per finished epoch), or [] if there is none."""
    if not TRAINING_LOG.exists():
        return []
    with TRAINING_LOG.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
    ap.add_argument("--augment-factor", type=int, default=5,
                    help="augmented copies added per training sample (0 = off)")
    ap.add_argument("--architecture", choices=["cnn_lstm", "cnn"], default="cnn_lstm",
                    help="cnn_lstm (default) keeps frame order for onset-consonant "
                         "words like cat/bat/hat; cnn is the faster order-agnostic original")
    ap.add_argument("--label-smoothing", type=float, default=0.05,
                    help="soften one-hot targets a little (helps with visually close classes)")
    ap.add_argument("--synthetic", action="store_true",
                    help="train on generated noise (pipeline smoke test, no dataset)")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from its best checkpoint, up to --epochs "
                         "in total (use the same data / --augment-factor / --seed as before)")
    args = ap.parse_args()

    # stdout is block-buffered when redirected to a file; keep epoch lines live
    sys.stdout.reconfigure(line_buffering=True)

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

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        # take VRAM as needed instead of all 4 GB up front (laptop GPU is shared with the display)
        tf.config.experimental.set_memory_growth(g, True)
    print("[device] " + (f"GPU: {', '.join(g.name for g in gpus)}" if gpus
                         else "CPU only - no CUDA GPU found (see requirements-gpu.txt)"))

    class SaveClassNamesWithCheckpoint(tf.keras.callbacks.Callback):
        """Write class_names.json only when ModelCheckpoint has written new weights, so a
        run stopped before its first checkpoint can't pair new names with old weights."""

        def __init__(self):
            super().__init__()
            self._mtime = WEIGHTS_PATH.stat().st_mtime if WEIGHTS_PATH.exists() else None

        def on_epoch_end(self, epoch, logs=None):
            mtime = WEIGHTS_PATH.stat().st_mtime if WEIGHTS_PATH.exists() else None
            if mtime != self._mtime:
                save_class_names(class_names)
                self._mtime = mtime

    initial_epoch, best_so_far = 0, None
    if args.resume:
        log_rows = read_training_log()
        if not WEIGHTS_PATH.exists() or not log_rows:
            sys.exit("--resume: no checkpoint / training_log.csv to resume from - run without it")
        saved_names = json.loads((MODEL_DIR / "class_names.json").read_text(encoding="utf-8"))
        if saved_names != class_names:
            sys.exit(f"--resume: checkpoint is for {saved_names}, dataset has {class_names}")
        initial_epoch = len(log_rows)
        best_so_far = max(float(r["val_accuracy"]) for r in log_rows if r.get("val_accuracy"))
        if initial_epoch >= args.epochs:
            sys.exit(f"--resume: already did {initial_epoch} epochs - raise --epochs to train more")
        # the checkpoint is the BEST epoch so far (save_best_only), not necessarily the last
        model = tf.keras.models.load_model(WEIGHTS_PATH)
        print(f"[resume] from {WEIGHTS_PATH.name}: {initial_epoch} epochs done, "
              f"best val_accuracy {best_so_far:.3f} -> continuing to epoch {args.epochs}")
    else:
        model = compile_model(
            build_model(num_classes, architecture=args.architecture),
            learning_rate=args.lr,
            label_smoothing=args.label_smoothing,
        )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(WEIGHTS_PATH), monitor="val_accuracy", mode="max",
            save_best_only=True, save_weights_only=False, verbose=1,
            # on --resume, only overwrite the checkpoint if we beat the earlier best
            initial_value_threshold=best_so_far,
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
    # per-epoch log written as it goes - long CPU runs can be watched, and a run that
    # gets stopped early still leaves its numbers behind
    callbacks += [
        SaveClassNamesWithCheckpoint(),
        tf.keras.callbacks.CSVLogger(str(TRAINING_LOG), append=args.resume),
    ]

    class ArrayBatches(tf.keras.utils.Sequence):
        """Hand the training arrays to Keras one batch at a time. Passing the numpy arrays
        straight to fit() makes TF copy the whole augmented set into one tensor first, which
        doubles RAM use - that ran the 8 GB laptop out of memory once training moved to GPU
        and the augmented set grew. Class weights go in as per-sample weights."""

        def __init__(self, x, y, weights, batch, seed):
            self.x, self.y, self.w, self.batch = x, y, weights, batch
            self.rng = np.random.default_rng(seed)
            self.order = self.rng.permutation(len(x))

        def __len__(self):
            return int(np.ceil(len(self.x) / self.batch))

        def __getitem__(self, i):
            idx = np.sort(self.order[i * self.batch:(i + 1) * self.batch])
            return self.x[idx], self.y[idx], self.w[idx]

        def on_epoch_end(self):
            self.order = self.rng.permutation(len(self.x))

    sample_weight = np.array([class_weight[c] for c in y_tr], dtype="float32")
    hist = model.fit(
        ArrayBatches(X_tr, y_tr_oh, sample_weight, args.batch, args.seed),
        validation_data=val_data,
        validation_batch_size=args.batch,  # default would be 32, too big for a 4 GB laptop GPU
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=2,
    )

    if args.resume:
        # hist only has the resumed epochs - take the whole run from the CSV instead
        rows = read_training_log()
        history = {k: [float(r[k]) for r in rows] for k in rows[0] if k != "epoch"}
    else:
        history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    (OUTPUT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    try:
        plot_history(history, OUTPUT_DIR / "history.png")
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not plot history: {exc}")

    if not WEIGHTS_PATH.exists():
        model.save(WEIGHTS_PATH)
    save_class_names(class_names)

    best = max(history.get("val_accuracy", history["accuracy"]))
    print(f"\n[done] best {'val_' if val_data else ''}accuracy: {best:.3f}")
    print(f"  weights -> {WEIGHTS_PATH}")
    print(f"  history -> {OUTPUT_DIR / 'history.json'}")


if __name__ == "__main__":
    main()
