"""
Week 7 - build the .npy dataset from recorded video clips.

Dhruv Sharma, sprint task: "Dataset preparation - prepared labeled data for training."

Alternative to live collection (collect.py): if you have one short clip per spoken
word, drop them in a folder tree by label and this turns each clip into a training
sample using the SAME face-detection + shared-preprocess path as everything else.

    raw_videos/
      hello/  clip1.mp4  clip2.mp4 ...
      dog/    clip1.mp4 ...

    python team_video_processing/data_collection/build_dataset.py --raw raw_videos

Output:
    team_video_processing/dataset/<word>/<id>.npy    (SEQ_LEN, 80, 112, 3) float32
    team_ai_model/model/class_names.json             sorted label list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
TEAM_VP = HERE.parent
REPO_ROOT = TEAM_VP.parent
sys.path.insert(0, str(TEAM_VP))

from preprocessing.preprocess import crop_lip_region, resize_lip, normalize_sequence  # noqa: E402
from face_detection.detector import FaceLandmarkDetector  # noqa: E402

DATASET_DIR = TEAM_VP / "dataset"
CLASS_NAMES_PATH = REPO_ROOT / "team_ai_model" / "model" / "class_names.json"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def lip_frames_from_video(path: Path, detector: FaceLandmarkDetector) -> list[np.ndarray]:
    """Every frame of the clip -> a resized lip crop (frames with no face are skipped)."""
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = detector.detect(frame)
        if res is None:
            continue
        crop = crop_lip_region(frame, res.landmarks)
        if crop is not None and crop.size:
            frames.append(resize_lip(crop))
    cap.release()
    return frames


def next_index(word_dir: Path) -> int:
    existing = [int(p.stem) for p in word_dir.glob("*.npy") if p.stem.isdigit()]
    return max(existing) + 1 if existing else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the LipSense .npy dataset from videos")
    ap.add_argument("--raw", required=True, help="folder with <word>/<clip>.<ext> subfolders")
    ap.add_argument("--min-frames", type=int, default=6, help="skip clips shorter than this")
    args = ap.parse_args()

    raw_root = Path(args.raw)
    if not raw_root.is_dir():
        sys.exit(f"not a folder: {raw_root}")

    detector = FaceLandmarkDetector()
    word_dirs = sorted(p for p in raw_root.iterdir() if p.is_dir())
    if not word_dirs:
        sys.exit(f"no <word>/ subfolders in {raw_root}")

    total_ok = total_skip = 0
    for wd in word_dirs:
        word = wd.name.strip().lower()
        clips = sorted(p for p in wd.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if not clips:
            continue
        out_dir = DATASET_DIR / word
        out_dir.mkdir(parents=True, exist_ok=True)

        for clip in clips:
            frames = lip_frames_from_video(clip, detector)
            if len(frames) < args.min_frames:
                print(f"  [skip] {word}/{clip.name}: only {len(frames)} usable frames")
                total_skip += 1
                continue
            tensor = normalize_sequence(frames)  # -> (SEQ_LEN, 80, 112, 3)
            idx = next_index(out_dir)
            np.save(out_dir / f"{idx:03d}.npy", tensor)
            print(f"  [ok]   {word}/{clip.name} -> dataset/{word}/{idx:03d}.npy "
                  f"({len(frames)} frames)")
            total_ok += 1

    classes = sorted(
        p.name for p in DATASET_DIR.iterdir()
        if p.is_dir() and any(p.glob("*.npy"))
    )
    CLASS_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASS_NAMES_PATH.write_text(json.dumps(classes, indent=2), encoding="utf-8")

    print(f"\ndone: {total_ok} samples written, {total_skip} skipped")
    print(f"classes ({len(classes)}): {classes}")
    print(f"class_names.json -> {CLASS_NAMES_PATH}")


if __name__ == "__main__":
    main()
