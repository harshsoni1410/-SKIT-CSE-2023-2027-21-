"""
LipSense - find the right --camera index for collect.py / predict_live.py.

Useful once a phone-as-webcam app (DroidCam, Iriun Webcam, etc.) is installed: Windows
now has more than one camera device and OpenCV's index numbering doesn't always match
what you'd expect from the Camera app.

Windows has several camera backends (Media Foundation, DirectShow, ...) and a virtual
webcam (DroidCam, Iriun) often only registers with DirectShow - OpenCV's default backend
on Windows won't always see it even though it shows up fine in the Windows Camera app.
So this tries EVERY index with BOTH the default backend and DirectShow explicitly.

Run:
    python team_video_processing/list_cameras.py

For each (index, backend) that actually opens, it shows a live preview window. Press:
    n  - try the next one
    q  - quit (prints the exact flags you need, e.g. --camera 2 --backend dshow)
"""

from __future__ import annotations

import cv2

BACKENDS = [
    ("default", cv2.CAP_ANY),
    ("dshow", cv2.CAP_DSHOW),
]


def open_camera(index: int, backend_name: str = "default"):
    """Shared by this script and collect.py / predict_live.py."""
    backend = dict(BACKENDS)[backend_name]
    return cv2.VideoCapture(index, backend)


def main() -> None:
    tried_any = False
    for idx in range(6):
        for backend_name, backend in BACKENDS:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue

            tried_any = True
            label = f"--camera {idx} --backend {backend_name}"
            print(f"\n{label} opened. Press 'n' for next, 'q' to stop here.")
            skip_rest_of_this_backend = False
            while True:
                ok, frame = cap.read()
                if not ok:
                    print(f"  {label} opened but returned no frames - skipping.")
                    break
                cv2.putText(frame, f"{label}   (n=next, q=use this one)",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("LipSense - find camera index", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("n"):
                    break
                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    print(f"\nUse:  {label}")
                    return
            cap.release()

    cv2.destroyAllWindows()
    if not tried_any:
        print("\nNo camera indices responded at all (0-5, default + dshow backends).")
    else:
        print("\nWent through every camera that opened - none was picked with 'q'.")
    print("If your phone-cam app shows up in the Windows Camera app but not here, try "
          "closing other apps that might be holding the camera open (Camera app, Zoom, "
          "Teams, OBS), then run this again.")


if __name__ == "__main__":
    main()
