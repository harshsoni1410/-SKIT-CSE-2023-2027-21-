"""
LipSense - find the right --camera index for collect.py / predict_live.py.

Useful once a phone-as-webcam app (DroidCam, Iriun Webcam, etc.) is installed: Windows
now has more than one camera device and OpenCV's index numbering doesn't always match
what you'd expect from the Camera app.

Run:
    python team_video_processing/list_cameras.py

For each index 0..5 that actually opens, it shows a live preview window with the index
number on screen. Press:
    n  - try the next index
    q  - quit (prints which index you were looking at, so you know which to use)
"""

from __future__ import annotations

import cv2


def main() -> None:
    for idx in range(6):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue

        print(f"\n--camera {idx} opened. Press 'n' for next, 'q' to stop here.")
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"  --camera {idx} opened but returned no frames - skipping.")
                break
            cv2.putText(frame, f"--camera {idx}   (n=next, q=use this one)",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("LipSense - find camera index", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("n"):
                break
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print(f"\nUse:  --camera {idx}")
                return
        cap.release()

    cv2.destroyAllWindows()
    print("\nNo more camera indices responded. If your phone-cam app is running, make "
          "sure its Windows client is connected before running this script.")


if __name__ == "__main__":
    main()
