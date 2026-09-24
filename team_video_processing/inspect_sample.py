"""
Inspect a recorded dataset sample - the #1 debugging step when predictions look random.

If the model predicts nonsense, first check the DATA: open a few .npy samples and make
sure each shows a clear, centred mouth moving through the word. If the crops are off,
dark, or barely show lips, no model will learn from them.

    python team_video_processing/inspect_sample.py                       # a random sample
    python team_video_processing/inspect_sample.py dataset/hello/003.npy # a specific one
    python team_video_processing/inspect_sample.py --all hello           # montage per word
    python team_video_processing/inspect_sample.py --summary             # quality table, all words

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

# same thresholds as collect.py's quality gate (kept here so this script needs no dlib import)
MIN_MOTION = 1.5
MIN_BRIGHTNESS = 25


def summarize_dataset() -> None:
    """Print per-word sample count + how many samples look too dark or static."""
    word_dirs = sorted(d for d in DATASET_DIR.iterdir() if d.is_dir() and any(d.glob("*.npy")))
    if not word_dirs:
        sys.exit(f"no .npy samples under {DATASET_DIR} - record some with collect.py first")

    print(f"{'word':<8}{'samples':>8}{'dark':>6}{'static':>8}{'brightness':>12}{'motion':>8}")
    for d in word_dirs:
        brightness, motion = [], []
        dark = static = 0
        for f in sorted(d.glob("*.npy")):
            seq = np.load(f) * 255.0
            b = float(seq.mean())
            m = float(np.abs(np.diff(seq, axis=0)).mean())
            brightness.append(b)
            motion.append(m)
            dark += b < MIN_BRIGHTNESS
            static += m < MIN_MOTION
        print(f"{d.name:<8}{len(brightness):>8}{dark:>6}{static:>8}"
              f"{np.mean(brightness):>12.1f}{np.mean(motion):>8.2f}")


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
    ap.add_argument("--summary", action="store_true",
                    help="print sample count and dark/static counts for every word (no files written)")
    args = ap.parse_args()

    if args.summary:
        summarize_dataset()
        return

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
