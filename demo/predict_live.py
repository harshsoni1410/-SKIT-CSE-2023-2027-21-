"""
Week 7 - LipSense standalone live demo (no browser).

Whole team - real-time prediction integration.

    webcam -> dlib face + 68 landmarks -> lip ROI crop -> speaking detection
           -> 22-frame sequence -> shared preprocess -> 3D CNN -> word + confidence

Run from the repo root (so relative model paths resolve):

    python demo/predict_live.py
    python demo/predict_live.py --camera 1 --flip

Keys:
    c  - calibrate the "mouth closed" baseline (keep mouth closed, then press c)
    q  - quit

If no trained model exists yet it still runs the capture pipeline and prints
"no model" instead of a word - useful for checking the camera + detection.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "team_video_processing"))

from preprocessing.preprocess import (  # noqa: E402
    SEQ_LEN, FRAME_W, FRAME_H,
    crop_lip_region, resize_lip, inner_lip_distance, normalize_sequence,
)
from face_detection.detector import FaceLandmarkDetector, draw_detection  # noqa: E402
from team_ai_model.training.predict import load_reader  # noqa: E402

# utterance-detection tuning (same values as data_collection/collect.py)
CALIB_FRAMES = 40
SILENCE_FRAMES = 7
MIN_UTTER_FRAMES = 6
MAX_UTTER_FRAMES = 60
LOW_CONFIDENCE = 0.6  # below this -> "uncertain" (PRD.md FR-10)


def draw_hud(frame, lines):
    y = 26
    for text, color in lines:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += 30


def main() -> None:
    ap = argparse.ArgumentParser(description="LipSense live demo")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--flip", action="store_true", help="mirror the webcam image")
    args = ap.parse_args()

    detector = FaceLandmarkDetector()
    reader = load_reader()
    if reader is None:
        print("[demo] no trained model - running capture only "
              "(train: python -m team_ai_model.training.train --epochs 40)")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"cannot open camera {args.camera}")

    calib_buffer: deque[float] = deque(maxlen=CALIB_FRAMES)
    calibrating = True
    open_threshold = close_threshold = None

    recording = False
    utter_frames: list[np.ndarray] = []
    silence = 0

    last_word = "-"
    last_conf = 0.0

    print("Keep your mouth CLOSED and press 'c' to calibrate. Then speak a word.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.flip:
            frame = cv2.flip(frame, 1)

        result = detector.detect(frame)
        state, color = "NO FACE", (0, 0, 255)

        if result is not None:
            lm = result.landmarks
            ratio = inner_lip_distance(lm)
            crop = crop_lip_region(frame, lm)
            lip = resize_lip(crop) if crop is not None and crop.size else None
            frame = draw_detection(frame, result)

            if calibrating:
                state, color = "CALIBRATING - mouth closed, press 'c'", (0, 200, 255)
                calib_buffer.append(ratio)
            elif lip is not None:
                speaking = ratio > (close_threshold if recording else open_threshold)
                if speaking:
                    if not recording:
                        recording, utter_frames, silence = True, [], 0
                    utter_frames.append(lip.copy())
                    silence = 0
                    state, color = f"RECORDING ({len(utter_frames)})", (0, 0, 255)
                    if len(utter_frames) >= MAX_UTTER_FRAMES:
                        recording = False
                elif recording:
                    silence += 1
                    state, color = "RECORDING (trailing)", (0, 140, 255)
                    if silence >= SILENCE_FRAMES:
                        recording = False
                else:
                    state, color = "NOT TALKING", (0, 255, 0)

                if not recording and utter_frames:
                    if len(utter_frames) >= MIN_UTTER_FRAMES:
                        seq = normalize_sequence(utter_frames)  # (SEQ_LEN, H, W, 3)
                        if reader is not None:
                            word, conf, _ = reader.predict(seq[None, ...])
                            last_word = word if conf >= LOW_CONFIDENCE else f"{word}?"
                            last_conf = conf
                            print(f"  -> {word}  ({conf:.2f})"
                                  + ("" if conf >= LOW_CONFIDENCE else "  [uncertain]"))
                        else:
                            last_word, last_conf = "no model", 0.0
                    utter_frames, silence = [], 0

        hud = [
            (f"PREDICTION: {last_word.upper()}", (255, 255, 255)),
            (f"confidence: {last_conf:.2f}", (200, 200, 200)),
            (state, color),
            ("keys: c=calibrate  q=quit", (160, 160, 160)),
        ]
        draw_hud(frame, hud)
        cv2.imshow("LipSense - live demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c") and len(calib_buffer) >= CALIB_FRAMES // 2:
            arr = np.array(calib_buffer)
            spread = max(0.04, 2.5 * float(arr.std()))
            open_threshold = float(arr.mean()) + spread
            close_threshold = float(arr.mean()) + spread * 0.6
            calibrating = False
            print(f"calibrated: open>{open_threshold:.3f}  close>{close_threshold:.3f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
