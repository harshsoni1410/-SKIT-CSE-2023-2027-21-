"""
Inspect a recorded dataset sample - the #1 debugging step when predictions look random.

If the model predicts nonsense, first check the DATA: open a few .npy samples and make
sure each shows a clear, centred mouth moving through the word. If the crops are off,
dark, or barely show lips, no model will learn from them.

    python team_video_processing/inspect_sample.py                       # a random sample
    python team_video_processing/inspect_sample.py dataset/hello/003.npy # a specific one
    python team_video_processing/inspect_sample.py --all hello           # montage per word

Saves a PNG montage (22 frames in a grid) next to this script, and also tries to show it.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "dataset"


def montage(seq: np.ndarray, cols: int = 8) -> np.ndarray:
    """(T, H, W, 3) 0..1 -> one (rows*H, cols*W, 3) uint8 image."""
    t, h, w, _ = seq.shape
    rows = (t + cols - 1) // cols
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i in range(t):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = (seq[i] * 255).clip(0, 255).astype("uint8")
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect a LipSense .npy sample")
    ap.add_argument("path", nargs="?", help="path to a .npy sample")
    ap.add_argument("--all", metavar="WORD", help="montage the first sample of each class, "
                                                  "or all samples of WORD")
    ap.add_argument("--show", action="store_true", help="also open a window (else just save the PNG)")
    args = ap.parse_args()

    import cv2

    if args.all:
        word_dir = DATASET_DIR / args.all
        files = sorted(word_dir.glob("*.npy"))
        if not files:
            sys.exit(f"no samples in {word_dir}")
        rows = []
        for f in files[:12]:
            rows.append(montage(np.load(f), cols=22))
        img = np.vstack(rows)
        out = HERE / f"_inspect_{args.all}.png"
    else:
        if args.path:
            path = Path(args.path)
            if not path.is_absolute():
                path = HERE / path if (HERE / path).exists() else Path.cwd() / path
        else:
            all_npy = list(DATASET_DIR.rglob("*.npy"))
            if not all_npy:
                sys.exit(f"no .npy files under {DATASET_DIR} - record some with collect.py first")
            path = random.choice(all_npy)
        seq = np.load(path)
        print(f"{path}\n  shape {seq.shape}  dtype {seq.dtype}  range ({seq.min():.2f}, {seq.max():.2f})")
        if seq.min() < -0.01 or seq.max() > 1.01:
            print("  [warn] values outside 0..1 - preprocessing issue")
        img = montage(seq)
        out = HERE / f"_inspect_{path.parent.name}_{path.stem}.png"

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out), img_bgr)
    print(f"saved -> {out}   (open it to check the mouth is centred and clear)")
    if args.show:
        cv2.imshow("sample (press any key)", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
