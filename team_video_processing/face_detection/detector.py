"""
Week 2 — LipSense face + landmark detection.

Dhruv Sharma, sprint task 3: "Face detection — implement face detection from video frames."

This wraps dlib's frontal-face detector and 68-point landmark predictor into one small
class so the rest of the project (data collection, the demo, evaluation) does not repeat
the same loading / detection code.

    detector = FaceLandmarkDetector()          # loads dlib models once
    result = detector.detect(frame_bgr)        # -> DetectionResult or None
    if result:
        result.face          # dlib rectangle (x1, y1, x2, y2 via .left()/.top()/...)
        result.landmarks     # (68, 2) int32 numpy array of (x, y)

Lip-region cropping stays in the shared preprocessing module
(`team_video_processing/preprocessing/preprocess.py`) — this file only finds the face
and the landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from preprocessing.preprocess import landmarks_to_np

# dlib's shape_predictor_68_face_landmarks.dat, renamed and stored under team_ai_model/
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTOR_PATH = REPO_ROOT / "team_ai_model" / "model" / "face_weights.dat"


@dataclass
class DetectionResult:
    """One face + its 68 landmarks for a single frame."""
    face: object                 # dlib.rectangle
    landmarks: np.ndarray        # (68, 2) int32

    @property
    def box(self) -> tuple[int, int, int, int]:
        """(x1, y1, x2, y2) bounding box of the face."""
        r = self.face
        return r.left(), r.top(), r.right(), r.bottom()

    @property
    def mouth_points(self) -> np.ndarray:
        """The 20 mouth landmarks (dlib indices 48–67)."""
        return self.landmarks[48:68]


class FaceLandmarkDetector:
    """Loads the dlib models once and detects the largest face per frame."""

    def __init__(self, predictor_path: str | Path | None = None, upsample: int = 0):
        try:
            import dlib
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "dlib is not installed. Run:  pip install dlib\n"
                "(dlib==20.0.1 ships a prebuilt Windows wheel for Python 3.10)"
            ) from exc

        path = Path(predictor_path) if predictor_path else DEFAULT_PREDICTOR_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Landmark model not found: {path}\n"
                "Download 'shape_predictor_68_face_landmarks.dat', rename it to "
                "'face_weights.dat' and put it in team_ai_model/model/."
            )

        self._dlib = dlib
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(str(path))
        self._upsample = upsample

    # ------------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray) -> DetectionResult | None:
        """
        Detect the largest face in a BGR frame and return its landmarks.
        Returns None if no face is found.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._detector(gray, self._upsample)
        if not faces:
            return None

        face = max(faces, key=lambda r: r.width() * r.height())
        shape = self._predictor(gray, face)
        landmarks = landmarks_to_np(shape)
        return DetectionResult(face=face, landmarks=landmarks)

    # ------------------------------------------------------------------
    def detect_all(self, frame_bgr: np.ndarray) -> list[DetectionResult]:
        """Detect every face in the frame (largest first)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = sorted(
            self._detector(gray, self._upsample),
            key=lambda r: r.width() * r.height(),
            reverse=True,
        )
        out = []
        for face in faces:
            shape = self._predictor(gray, face)
            out.append(DetectionResult(face=face, landmarks=landmarks_to_np(shape)))
        return out


# ----------------------------------------------------------------------
def draw_detection(frame_bgr: np.ndarray, result: DetectionResult,
                   box_color=(0, 200, 255), point_color=(255, 200, 0)) -> np.ndarray:
    """Draw the face box + mouth landmarks on a copy of the frame (for debugging)."""
    out = frame_bgr.copy()
    x1, y1, x2, y2 = result.box
    cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
    for (x, y) in result.mouth_points:
        cv2.circle(out, (int(x), int(y)), 1, point_color, -1)
    return out


# quick self-test (needs a webcam + the dlib model)
if __name__ == "__main__":  # pragma: no cover
    import sys

    det = FaceLandmarkDetector()
    cap = cv2.VideoCapture(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    if not cap.isOpened():
        raise SystemExit("cannot open camera")

    print("press q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = det.detect(frame)
        if res is not None:
            frame = draw_detection(frame, res)
            cv2.putText(frame, "FACE", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "NO FACE", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("face_detection self-test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
