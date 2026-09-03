"""
Week 4 - dataset loading for LipSense training.

Harsh Soni, sprint task: model development (data pipeline).

Reads the word-wise .npy dataset produced by the video team:

    team_video_processing/dataset/<word>/<id>.npy      shape (22, 80, 112, 3), float32, 0..1

Rules (docs/team_plan.md "Shared rules"):
  - class order = sorted(folder names), never hardcoded
  - the same order is written to class_names.json so training and the demo agree
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import INPUT_SHAPE

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "team_video_processing" / "dataset"
CLASS_NAMES_PATH = REPO_ROOT / "team_ai_model" / "model" / "class_names.json"


def list_classes(dataset_dir: str | Path = DATASET_DIR) -> list[str]:
    """Word labels = sorted names of dataset subfolders that contain at least one .npy."""
    d = Path(dataset_dir)
    if not d.exists():
        return []
    return sorted(
        p.name for p in d.iterdir()
        if p.is_dir() and any(p.glob("*.npy"))
    )


def load_dataset(dataset_dir: str | Path = DATASET_DIR
                 ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load every sample into memory.
    Returns (X, y, class_names):
        X -> (N, 22, 80, 112, 3) float32
        y -> (N,) int64  class indices
    """
    classes = list_classes(dataset_dir)
    if not classes:
        raise FileNotFoundError(
            f"No class folders with .npy files under {dataset_dir}.\n"
            "Record samples with team_video_processing/data_collection/collect.py, "
            "or run training with --synthetic for a pipeline smoke test."
        )

    X: list[np.ndarray] = []
    y: list[int] = []
    for idx, cls in enumerate(classes):
        for f in sorted((Path(dataset_dir) / cls).glob("*.npy")):
            arr = np.load(f)
            if arr.shape != INPUT_SHAPE:
                print(f"  [skip] {f.name}: shape {arr.shape} != {INPUT_SHAPE}")
                continue
            X.append(arr.astype("float32"))
            y.append(idx)

    if not X:
        raise ValueError("Found class folders but no samples with the expected shape.")

    return np.stack(X), np.asarray(y, dtype="int64"), classes


def train_val_split(X: np.ndarray, y: np.ndarray, val_frac: float = 0.2, seed: int = 42
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split - each word keeps the same train/val ratio."""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac))) if len(idx) > 1 else 0
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def to_onehot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(y), num_classes), dtype="float32")
    out[np.arange(len(y)), y] = 1.0
    return out


def save_class_names(class_names: list[str], path: str | Path = CLASS_NAMES_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    return path


def load_class_names(path: str | Path = CLASS_NAMES_PATH) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_synthetic(num_classes: int = 3, per_class: int = 8, seed: int = 0
                   ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Fake dataset for a pipeline smoke test - random noise plus a small per-class
    brightness offset so the model can actually fit it in a couple of epochs.
    """
    rng = np.random.default_rng(seed)
    n = num_classes * per_class
    X = rng.random((n, *INPUT_SHAPE), dtype="float32") * 0.5
    y = np.repeat(np.arange(num_classes), per_class).astype("int64")
    for cls in range(num_classes):
        X[y == cls] += 0.06 * cls
    np.clip(X, 0.0, 1.0, out=X)
    classes = [f"word{i}" for i in range(num_classes)]
    return X, y, classes


if __name__ == "__main__":
    print("dataset dir:", DATASET_DIR)
    found = list_classes()
    print("classes found:", found or "(none - use --synthetic for training)")
    Xs, ys, cs = make_synthetic()
    Xtr, ytr, Xva, yva = train_val_split(Xs, ys)
    print(f"synthetic: X{Xs.shape} classes={cs}  train={len(Xtr)} val={len(Xva)}")
