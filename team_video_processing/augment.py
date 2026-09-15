"""
Week 4 - data augmentation for LipSense.

Dhruv Sharma, sprint task: "Data normalization / dataset preparation."

Works on a single utterance tensor:

    seq : np.ndarray, shape (SEQ_LEN, 80, 112, 3), float32, values 0..1

The same transform is applied to every frame of a sequence so lip motion stays
consistent across time. Approved augmentations (team_video_processing/README.md):
horizontal flip, brightness jitter, small spatial shift.

Augmentation is training-only - never applied to live demo frames.
"""

from __future__ import annotations

import numpy as np

from preprocessing.preprocess import SEQ_LEN, FRAME_H, FRAME_W  # noqa: F401  (contract check)


def horizontal_flip(seq: np.ndarray) -> np.ndarray:
    """Mirror left<->right. Fine for word-level lip reading (mouth is ~symmetric)."""
    return seq[:, :, ::-1, :].copy()


def brightness_jitter(seq: np.ndarray, max_delta: float = 0.15,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Add one random brightness offset to the whole sequence."""
    rng = rng or np.random.default_rng()
    delta = rng.uniform(-max_delta, max_delta)
    return np.clip(seq + delta, 0.0, 1.0).astype("float32")


def contrast_jitter(seq: np.ndarray, max_factor: float = 0.15,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    factor = 1.0 + rng.uniform(-max_factor, max_factor)
    mean = seq.mean()
    return np.clip((seq - mean) * factor + mean, 0.0, 1.0).astype("float32")


def time_warp(seq: np.ndarray, max_speed: float = 0.15,
             rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Resample the time axis at a random speed (0.85x-1.15x by default) then resample back
    to SEQ_LEN. Simulates the same speaker saying the word a bit faster/slower - useful
    because the model needs to recognise the initial-consonant mouth shape regardless of
    exactly how many frames it lasted, not memorise one fixed timing.
    """
    rng = rng or np.random.default_rng()
    speed = 1.0 + rng.uniform(-max_speed, max_speed)
    n = seq.shape[0]
    src_idx = np.clip(np.arange(n) * speed, 0, n - 1)
    lo = np.floor(src_idx).astype(int)
    hi = np.clip(lo + 1, 0, n - 1)
    frac = (src_idx - lo).reshape(-1, 1, 1, 1)
    warped = seq[lo] * (1 - frac) + seq[hi] * frac
    return warped.astype("float32")


def spatial_shift(seq: np.ndarray, max_shift: int = 6,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Shift every frame by the same small (dx, dy), padding with the edge pixels."""
    rng = rng or np.random.default_rng()
    dx = int(rng.integers(-max_shift, max_shift + 1))
    dy = int(rng.integers(-max_shift, max_shift + 1))
    shifted = np.roll(seq, shift=(dy, dx), axis=(1, 2))
    if dy > 0:
        shifted[:, :dy, :, :] = shifted[:, dy:dy + 1, :, :]
    elif dy < 0:
        shifted[:, dy:, :, :] = shifted[:, dy - 1:dy, :, :]
    if dx > 0:
        shifted[:, :, :dx, :] = shifted[:, :, dx:dx + 1, :]
    elif dx < 0:
        shifted[:, :, dx:, :] = shifted[:, :, dx - 1:dx, :]
    return shifted.astype("float32")


def augment(seq: np.ndarray, rng: np.random.Generator | None = None,
            p_flip: float = 0.5, p_bright: float = 0.7,
            p_contrast: float = 0.4, p_shift: float = 0.6,
            p_timewarp: float = 0.5) -> np.ndarray:
    """Apply a random combination of transforms to one sequence."""
    rng = rng or np.random.default_rng()
    out = seq.astype("float32", copy=True)
    if rng.random() < p_flip:
        out = horizontal_flip(out)
    if rng.random() < p_timewarp:
        out = time_warp(out, rng=rng)
    if rng.random() < p_bright:
        out = brightness_jitter(out, rng=rng)
    if rng.random() < p_contrast:
        out = contrast_jitter(out, rng=rng)
    if rng.random() < p_shift:
        out = spatial_shift(out, rng=rng)
    return out


def augment_batch(X: np.ndarray, y: np.ndarray, factor: int = 1,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand a dataset: keep the originals and add `factor` augmented copies each.
    Returns (X_aug, y_aug) with len = len(X) * (1 + factor).
    """
    rng = np.random.default_rng(seed)
    extra_X, extra_y = [], []
    for _ in range(factor):
        for i in range(len(X)):
            extra_X.append(augment(X[i], rng=rng))
            extra_y.append(y[i])
    if not extra_X:
        return X, y
    return (np.concatenate([X, np.stack(extra_X)]),
            np.concatenate([y, np.asarray(extra_y, dtype=y.dtype)]))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fake = rng.random((SEQ_LEN, FRAME_H, FRAME_W, 3), dtype="float32")
    for name, fn in [("flip", horizontal_flip), ("bright", brightness_jitter),
                     ("contrast", contrast_jitter), ("shift", spatial_shift),
                     ("timewarp", time_warp), ("augment", augment)]:
        out = fn(fake)
        assert out.shape == fake.shape, name
        assert out.min() >= 0.0 and out.max() <= 1.0, name
        print(f"{name:9s} ok  shape={out.shape}  range=({out.min():.2f}, {out.max():.2f})")

    X = rng.random((6, SEQ_LEN, FRAME_H, FRAME_W, 3), dtype="float32")
    y = np.array([0, 0, 1, 1, 2, 2])
    Xa, ya = augment_batch(X, y, factor=2)
    print(f"augment_batch: {X.shape} -> {Xa.shape}, labels {ya.shape}")
