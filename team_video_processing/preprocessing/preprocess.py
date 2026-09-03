"""
LipSense - shared preprocessing module.

This single file is imported by data collection, training and the live demo. It guarantees that
frames are processed exactly the same way everywhere (see PRD.md section 7 - "Model input
contract").

If any constant here changes, the team + PRD.md must be updated, the dataset rebuilt and the
model retrained.
"""

from __future__ import annotations

import numpy as np
import cv2

# --------------------------------------------------------------------------------------
# MODEL INPUT CONTRACT  (training == prediction)
# --------------------------------------------------------------------------------------
SEQ_LEN = 22        # frames per utterance (fixed)
FRAME_H = 80        # lip crop height
FRAME_W = 112       # lip crop width
CHANNELS = 3        # 3 = RGB
COLOR_MODE = "rgb"  # OpenCV gives BGR -> we convert to RGB

# mouth points in the dlib 68-point landmark model
MOUTH_POINTS = list(range(48, 68))   # 48..67 -> outer + inner lip
INNER_TOP = 62                        # inner upper lip
INNER_BOTTOM = 66                     # inner lower lip
OUTER_LEFT = 48
OUTER_RIGHT = 54

# extra margin around the lip crop (fraction of box size)
CROP_MARGIN = 0.35


# --------------------------------------------------------------------------------------
# 1. LIP REGION CROP
# --------------------------------------------------------------------------------------
def landmarks_to_np(shape) -> np.ndarray:
    """dlib `shape` object -> (68, 2) numpy array of (x, y)."""
    coords = np.zeros((68, 2), dtype=np.int32)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def crop_lip_region(frame_bgr: np.ndarray, landmarks_np: np.ndarray) -> np.ndarray | None:
    """
    Cut only the mouth region out of the full frame using the 68 landmarks.
    Returns a BGR crop (not resized yet), or None if the box is invalid.
    """
    h, w = frame_bgr.shape[:2]
    mouth = landmarks_np[MOUTH_POINTS]

    x_min, y_min = mouth.min(axis=0)
    x_max, y_max = mouth.max(axis=0)

    bw = x_max - x_min
    bh = y_max - y_min
    if bw <= 0 or bh <= 0:
        return None

    mx = int(bw * CROP_MARGIN)
    my = int(bh * CROP_MARGIN)

    x1 = max(0, x_min - mx)
    y1 = max(0, y_min - my)
    x2 = min(w, x_max + mx)
    y2 = min(h, y_max + my)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame_bgr[y1:y2, x1:x2]


def resize_lip(crop_bgr: np.ndarray) -> np.ndarray:
    """Resize a lip crop to the model input size (FRAME_W x FRAME_H). Stays BGR, uint8."""
    return cv2.resize(crop_bgr, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------------------
# 2. SPEAKING DETECTION helper
# --------------------------------------------------------------------------------------
def inner_lip_distance(landmarks_np: np.ndarray) -> float:
    """
    Vertical distance between the inner upper lip (62) and inner lower lip (66),
    normalized by mouth width so distance from the camera does not matter.
    """
    top = landmarks_np[INNER_TOP]
    bottom = landmarks_np[INNER_BOTTOM]
    left = landmarks_np[OUTER_LEFT]
    right = landmarks_np[OUTER_RIGHT]

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right) + 1e-6
    return float(vertical / horizontal)


# --------------------------------------------------------------------------------------
# 3. SEQUENCE -> FIXED LENGTH
# --------------------------------------------------------------------------------------
def fix_sequence_length(frames: list[np.ndarray], seq_len: int = SEQ_LEN) -> list[np.ndarray]:
    """
    Force an utterance's frame list to exactly `seq_len` frames.
    - more frames  -> uniform sampling
    - fewer frames -> pad by repeating the last frame
    """
    n = len(frames)
    if n == 0:
        raise ValueError("fix_sequence_length: empty frame list")

    if n == seq_len:
        return frames
    if n > seq_len:
        idx = np.linspace(0, n - 1, seq_len).round().astype(int)
        return [frames[i] for i in idx]
    return frames + [frames[-1]] * (seq_len - n)


# --------------------------------------------------------------------------------------
# 4. FINAL TENSOR
# --------------------------------------------------------------------------------------
def normalize_sequence(frames: list[np.ndarray]) -> np.ndarray:
    """
    List of BGR uint8 lip crops (already resized to FRAME_W x FRAME_H)
      -> np.float32 array (SEQ_LEN, FRAME_H, FRAME_W, CHANNELS), values 0..1, RGB.
    """
    frames = fix_sequence_length(frames, SEQ_LEN)
    out = np.zeros((SEQ_LEN, FRAME_H, FRAME_W, CHANNELS), dtype=np.float32)
    for i, f in enumerate(frames):
        if f.shape[:2] != (FRAME_H, FRAME_W):
            f = cv2.resize(f, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        if COLOR_MODE == "rgb":
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        out[i] = f.astype(np.float32) / 255.0
    return out


def frames_to_tensor(frames: list[np.ndarray]) -> np.ndarray:
    """
    For live inference: list of BGR lip crops -> (1, SEQ_LEN, FRAME_H, FRAME_W, CHANNELS).
    Ready to pass straight into model.predict().
    """
    seq = normalize_sequence(frames)
    return np.expand_dims(seq, axis=0)


# quick self-test
if __name__ == "__main__":
    fake = [np.random.randint(0, 255, (90, 130, 3), dtype=np.uint8) for _ in range(10)]
    t = frames_to_tensor(fake)
    print("tensor shape:", t.shape, "dtype:", t.dtype, "range:", t.min(), t.max())
    assert t.shape == (1, SEQ_LEN, FRAME_H, FRAME_W, CHANNELS)
    print("OK")
