"""
LipSense - data collection tool.

Dhruv Sharma, first sprint task: "Collect video dataset". Build 9 - multi-word session
mode + automatic quality gate, so one recording session can cover the whole vocabulary
and bad samples (blinks, false triggers, dark frames) don't silently pollute training.

Records lip-region frame sequences from the webcam, one word at a time, with automatic
utterance detection (no need to press a key for every sample).

Pipeline per frame:
    webcam -> face detection (dlib) -> 68 landmarks (dlib) -> lip ROI crop -> resize
    -> speaking detection (inner-lip distance vs a calibrated threshold)

When an utterance is detected it is normalized to a fixed-length tensor by the SHARED
preprocessing module, quality-checked, and saved as a .npy file:

    team_video_processing/dataset/<word>/<index>.npy      shape (SEQ_LEN, 80, 112, 3), float32

A CSV log of every save/reject (with a timestamp) is kept at
team_video_processing/dataset/<word>/session_log.csv - useful to see how many different
sessions (lighting/time of day/distance from camera) went into a word, since recording
everything in one sitting makes the dataset overfit to that one sitting's conditions.

Run from anywhere. Single word:
    python team_video_processing/data_collection/collect.py --word hello --samples 20

Multiple words in one sitting (recommended - cycles through the list automatically):
    python team_video_processing/data_collection/collect.py --words cat,bat,hat,mat,rat,sat --samples 20

Using a phone as a webcam (DroidCam/Iriun) instead of the laptop's built-in camera -
find the right --camera/--backend combo first with team_video_processing/list_cameras.py,
then pass both here, e.g.:
    python team_video_processing/data_collection/collect.py --words ... --camera 2 --backend dshow

Keys while running:
    c  - (re)calibrate the "mouth closed" baseline (keep mouth closed, then press c)
    u  - undo / delete the last saved sample
    n  - skip to the next word early (multi-word mode)
    q  - quit
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
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


# ----------------------------------------------------------------------------------------
# quality gate - catches the two most common causes of bad samples that silently hurt
# accuracy: a false trigger (blink / camera noise, almost no motion) and a too-dark crop
# (lighting problem, the model sees mostly noise). Both pass MIN_UTTER_FRAMES easily.
# ----------------------------------------------------------------------------------------
MIN_MOTION = 1.5     # mean abs frame-to-frame pixel change (0-255 scale) below this = suspect
MIN_BRIGHTNESS = 25  # mean pixel value (0-255) below this = too dark


def check_quality(raw_frames: list[np.ndarray]) -> tuple[bool, str]:
    """raw_frames: list of uint8 BGR lip crops (pre-normalization). Returns (ok, reason)."""
    stacked = np.stack(raw_frames).astype(np.float32)
    brightness = float(stacked.mean())
    if brightness < MIN_BRIGHTNESS:
        return False, f"too dark (mean brightness {brightness:.0f} < {MIN_BRIGHTNESS})"

    diffs = np.abs(np.diff(stacked, axis=0)).mean()
    if diffs < MIN_MOTION:
        return False, f"almost no motion (mean frame diff {diffs:.2f} < {MIN_MOTION}) - likely a false trigger"

    return True, "ok"


def log_session_event(word_dir: Path, event: str, detail: str = "") -> None:
    log_path = word_dir / "session_log.csv"
    is_new = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "event", "detail"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), event, detail])


def draw_hud(frame, lines, color=(0, 255, 0)):
    y = 26
    for text, c in lines:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2, cv2.LINE_AA)
        y += 30


def record_word(cap, detector, predictor, word: str, samples: int, flip: bool,
                calib: dict) -> str:
    """
    Run the recording loop for a single word. `calib` is shared across words (a dict with
    closed_baseline/open_threshold/close_threshold/calib_buffer/calibrating) so calibration
    done for the first word carries over to the rest unless re-done with 'c'.

    Returns "quit", "next" (user pressed n / word finished) to tell the caller what to do.
    """
    word_dir = DATASET_DIR / word
    word_dir.mkdir(parents=True, exist_ok=True)

    recording = False
    utter_frames: list[np.ndarray] = []
    silence_count = 0

    collected = next_sample_index(word_dir)   # continue if some samples already exist
    last_saved_path: Path | None = None
    target = collected + samples

    print(f"\nRecording '{word}'  ->  {word_dir}")
    if calib["calibrating"]:
        print("Keep your mouth CLOSED and press 'c' to calibrate. Then speak the word normally.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if flip:
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
            if calib["calibrating"]:
                state_text, state_color = "CALIBRATING - keep mouth closed, press 'c'", (0, 200, 255)
                calib["calib_buffer"].append(ratio)
                if len(calib["calib_buffer"]) > CALIB_FRAMES:
                    calib["calib_buffer"].pop(0)

            # ---- speaking detection (only after calibration) ----
            elif calib["open_threshold"] is not None and lip_crop is not None:
                open_threshold = calib["open_threshold"]
                close_threshold = calib["close_threshold"]
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
                        ok_quality, reason = check_quality(utter_frames)
                        if ok_quality:
                            tensor = normalize_sequence(utter_frames)  # (SEQ_LEN, H, W, 3)
                            idx = next_sample_index(word_dir)
                            out_path = word_dir / f"{idx:03d}.npy"
                            np.save(out_path, tensor)
                            last_saved_path = out_path
                            collected = idx + 1
                            print(f"  saved {out_path.name}  ({len(utter_frames)} raw frames)")
                            log_session_event(word_dir, "saved", out_path.name)
                            state_text, state_color = "SAVED", (0, 255, 0)
                        else:
                            print(f"  rejected: {reason}")
                            log_session_event(word_dir, "rejected", reason)
                            state_text, state_color = "REJECTED - " + reason[:30], (0, 100, 255)
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
                        + (f"  open>{calib['open_threshold']:.3f}" if calib["open_threshold"] else "  (not calibrated)"),
                        (200, 200, 200)))
        hud.append(("keys: c=calibrate  u=undo  n=next word  q=quit", (160, 160, 160)))
        draw_hud(frame, hud)

        # show the current lip crop (top-right)
        if lip_crop is not None:
            preview = cv2.resize(lip_crop, (FRAME_W * 2, FRAME_H * 2), interpolation=cv2.INTER_NEAREST)
            h, w = preview.shape[:2]
            frame[10:10 + h, frame.shape[1] - w - 10:frame.shape[1] - 10] = preview

        cv2.imshow("LipSense - collect", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return "quit"
        elif key == ord("n"):
            print(f"Skipping to next word ({collected} samples collected for '{word}').")
            return "next"
        elif key == ord("c"):
            if len(calib["calib_buffer"]) >= CALIB_FRAMES // 2:
                arr = np.array(calib["calib_buffer"])
                closed_baseline = float(arr.mean())
                spread = max(0.04, 2.5 * float(arr.std()))
                calib["closed_baseline"] = closed_baseline
                calib["open_threshold"] = closed_baseline + spread
                calib["close_threshold"] = closed_baseline + spread * 0.6
                calib["calibrating"] = False
                print(f"calibrated: closed={closed_baseline:.3f}  open>{calib['open_threshold']:.3f}")
            else:
                print("not enough frames yet - look at the camera with mouth closed")
        elif key == ord("u"):
            if last_saved_path and last_saved_path.exists():
                last_saved_path.unlink()
                collected = max(0, collected - 1)
                print(f"deleted {last_saved_path.name}")
                log_session_event(word_dir, "undo", last_saved_path.name)
                last_saved_path = None
            else:
                print("nothing to undo")

        if collected >= target:
            print(f"Done - collected {collected} samples for '{word}'.")
            time.sleep(0.5)
            return "next"

    return "next"


def main():
    ap = argparse.ArgumentParser(description="LipSense webcam data collection")
    ap.add_argument("--word", help="single word / class label to record")
    ap.add_argument("--words", help="comma-separated list of words to record in one sitting, "
                                     "e.g. cat,bat,hat,mat,rat,sat")
    ap.add_argument("--samples", type=int, default=20, help="how many samples to collect per word")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--backend", choices=["default", "dshow"], default="default",
                    help="OpenCV capture backend - use 'dshow' for phone-as-webcam apps "
                         "(DroidCam/Iriun) if 'default' doesn't find them "
                         "(see team_video_processing/list_cameras.py)")
    ap.add_argument("--flip", action="store_true", help="mirror the webcam image")
    args = ap.parse_args()

    if args.words:
        word_list = [w.strip().lower() for w in args.words.split(",") if w.strip()]
    elif args.word:
        word_list = [args.word.strip().lower()]
    else:
        sys.exit("pass --word <word> or --words word1,word2,...")

    _dlib, detector, predictor = load_dlib()

    backend = cv2.CAP_DSHOW if args.backend == "dshow" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.camera} (backend={args.backend}). "
                  f"Run team_video_processing/list_cameras.py to find the right combo.")

    calib = {
        "closed_baseline": None,
        "open_threshold": None,
        "close_threshold": None,
        "calib_buffer": [],
        "calibrating": True,
    }

    print(f"Session plan: {word_list}  ({args.samples} samples each)")
    for word in word_list:
        result = record_word(cap, detector, predictor, word, args.samples, args.flip, calib)
        if result == "quit":
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nSession finished.")


if __name__ == "__main__":
    main()
