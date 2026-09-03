"""
LipSense - data collection tool.

Member 1, first sprint task: "Collect video dataset".

Records lip-region frame sequences from the webcam, one word at a time, with automatic
utterance detection (no need to press a key for every sample).

Pipeline per frame:
    webcam -> face detection (dlib) -> 68 landmarks (dlib) -> lip ROI crop -> resize
    -> speaking detection (inner-lip distance vs a calibrated threshold)

When an utterance is detected it is normalized to a fixed-length tensor by the SHARED
preprocessing module and saved as a .npy file:

    team_video_processing/dataset/<word>/<index>.npy      shape (SEQ_LEN, 80, 112, 3), float32

Run from anywhere:
    python team_video_processing/data_collection/collect.py --word hello --samples 20

Keys while running:
    c  - (re)calibrate the "mouth closed" baseline (keep mouth closed, then press c)
    u  - undo / delete the last saved sample
    q  - quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------------------
# make the shared preprocessing module importable regardless of the current directory
# ----------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent                 # .../team_video_processing/data_collection
TEAM_VP = HERE.parent                                  # .../team_video_processing
REPO = TEAM_VP.parent                                  # repo root
sys.path.insert(0, str(TEAM_VP))

from preprocessing.preprocess import (  # noqa: E402
    SEQ_LEN, FRAME_H, FRAME_W,
    landmarks_to_np, crop_lip_region, resize_lip,
    inner_lip_distance, normalize_sequence,
)

# dlib's 68-point predictor, stored (renamed) under team_ai_model/model/
PREDICTOR_PATH = REPO / "team_ai_model" / "model" / "face_weights.dat"
DATASET_DIR = TEAM_VP / "dataset"

# ----------------------------------------------------------------------------------------
# utterance-detection tuning
# ----------------------------------------------------------------------------------------
CALIB_FRAMES = 40        # frames used to measure the "mouth closed" baseline
SILENCE_FRAMES = 7       # consecutive "closed" frames that end an utterance
MIN_UTTER_FRAMES = 6     # shorter than this -> discard (likely a blink / noise)
MAX_UTTER_FRAMES = 60    # safety cap


def load_dlib():
    """Load the dlib face detector + landmark predictor, with a clear message if missing."""
    try:
        import dlib
    except ImportError:
        sys.exit(
            "dlib is not installed.\n"
            "  Try:  pip install dlib\n"
            "  (needs CMake + Visual Studio Build Tools on Windows, or a prebuilt wheel)\n"
        )
    if not PREDICTOR_PATH.exists():
        sys.exit(
            f"Landmark model not found: {PREDICTOR_PATH}\n"
            "Download 'shape_predictor_68_face_landmarks.dat', rename it to 'face_weights.dat'\n"
            f"and place it in: {PREDICTOR_PATH.parent}\n"
        )
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(PREDICTOR_PATH))
    return dlib, detector, predictor


def largest_face(faces):
    """Pick the biggest detected face (assume the speaker is closest to the camera)."""
    return max(faces, key=lambda r: r.width() * r.height())


def next_sample_index(word_dir: Path) -> int:
    existing = sorted(int(p.stem) for p in word_dir.glob("*.npy") if p.stem.isdigit())
    return (existing[-1] + 1) if existing else 0


def draw_hud(frame, lines, color=(0, 255, 0)):
    y = 26
    for text, c in lines:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2, cv2.LINE_AA)
        y += 30


def main():
    ap = argparse.ArgumentParser(description="LipSense webcam data collection")
    ap.add_argument("--word", required=True, help="word / class label to record")
    ap.add_argument("--samples", type=int, default=20, help="how many samples to collect")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--flip", action="store_true", help="mirror the webcam image")
    args = ap.parse_args()

    word = args.word.strip().lower()
    word_dir = DATASET_DIR / word
    word_dir.mkdir(parents=True, exist_ok=True)

    _dlib, detector, predictor = load_dlib()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.camera}")

    # ---- state ----
    closed_baseline = None          # calibrated "mouth closed" ratio
    open_threshold = None
    close_threshold = None
    calib_buffer: list[float] = []
    calibrating = True

    recording = False
    utter_frames: list[np.ndarray] = []
    silence_count = 0

    collected = next_sample_index(word_dir)   # continue if some samples already exist
    last_saved_path: Path | None = None
    target = collected + args.samples

    print(f"Recording '{word}'  ->  {word_dir}")
    print("Keep your mouth CLOSED and press 'c' to calibrate. Then speak the word normally.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.flip:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        state_text = "NO FACE"
        state_color = (0, 0, 255)
        ratio = None
        lip_crop = None

        if faces:
            face = largest_face(faces)
            shape = predictor(gray, face)
            lm = landmarks_to_np(shape)

            ratio = inner_lip_distance(lm)

            crop = crop_lip_region(frame, lm)
            if crop is not None and crop.size > 0:
                lip_crop = resize_lip(crop)

            # draw mouth landmarks
            for (x, y) in lm[48:68]:
                cv2.circle(frame, (int(x), int(y)), 1, (255, 200, 0), -1)

            # ---- calibration ----
            if calibrating:
                state_text, state_color = "CALIBRATING - keep mouth closed, press 'c'", (0, 200, 255)
                calib_buffer.append(ratio)
                if len(calib_buffer) > CALIB_FRAMES:
                    calib_buffer.pop(0)

            # ---- speaking detection (only after calibration) ----
            elif open_threshold is not None and lip_crop is not None:
                speaking = ratio > open_threshold if not recording else ratio > close_threshold

                if speaking:
                    if not recording:
                        recording = True
                        utter_frames = []
                        silence_count = 0
                    utter_frames.append(lip_crop.copy())
                    silence_count = 0
                    state_text, state_color = f"RECORDING WORD ({len(utter_frames)})", (0, 0, 255)

                    if len(utter_frames) >= MAX_UTTER_FRAMES:
                        recording = False  # force-close; will be handled below
                else:
                    if recording:
                        silence_count += 1
                        state_text, state_color = "RECORDING WORD (trailing)", (0, 140, 255)
                        if silence_count >= SILENCE_FRAMES:
                            recording = False
                    else:
                        state_text, state_color = "NOT TALKING", (0, 255, 0)

                # ---- utterance just ended ----
                if not recording and utter_frames:
                    if len(utter_frames) >= MIN_UTTER_FRAMES:
                        tensor = normalize_sequence(utter_frames)  # (SEQ_LEN, H, W, 3)
                        idx = next_sample_index(word_dir)
                        out_path = word_dir / f"{idx:03d}.npy"
                        np.save(out_path, tensor)
                        last_saved_path = out_path
                        collected = idx + 1
                        print(f"  saved {out_path.name}  ({len(utter_frames)} raw frames)")
                        state_text, state_color = "SAVED", (0, 255, 0)
                    else:
                        print(f"  discarded short utterance ({len(utter_frames)} frames)")
                    utter_frames = []
                    silence_count = 0

        # ---------------- HUD ----------------
        hud = [
            (f"WORD: {word}", (255, 255, 255)),
            (f"COLLECTED: {collected} / {target}", (255, 255, 255)),
            (state_text, state_color),
        ]
        if ratio is not None:
            hud.append((f"lip ratio: {ratio:.3f}"
                        + (f"  open>{open_threshold:.3f}" if open_threshold else "  (not calibrated)"),
                        (200, 200, 200)))
        hud.append(("keys: c=calibrate  u=undo  q=quit", (160, 160, 160)))
        draw_hud(frame, hud)

        # show the current lip crop (top-right)
        if lip_crop is not None:
            preview = cv2.resize(lip_crop, (FRAME_W * 2, FRAME_H * 2), interpolation=cv2.INTER_NEAREST)
            h, w = preview.shape[:2]
            frame[10:10 + h, frame.shape[1] - w - 10:frame.shape[1] - 10] = preview

        cv2.imshow("LipSense - collect", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("c"):
            if len(calib_buffer) >= CALIB_FRAMES // 2:
                arr = np.array(calib_buffer)
                closed_baseline = float(arr.mean())
                spread = max(0.04, 2.5 * float(arr.std()))
                open_threshold = closed_baseline + spread
                close_threshold = closed_baseline + spread * 0.6
                calibrating = False
                print(f"calibrated: closed={closed_baseline:.3f}  open>{open_threshold:.3f}")
            else:
                print("not enough frames yet - look at the camera with mouth closed")
        elif key == ord("u"):
            if last_saved_path and last_saved_path.exists():
                last_saved_path.unlink()
                collected = max(0, collected - 1)
                print(f"deleted {last_saved_path.name}")
                last_saved_path = None
            else:
                print("nothing to undo")

        if collected >= target:
            print(f"Done - collected {collected} samples for '{word}'.")
            time.sleep(0.5)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
