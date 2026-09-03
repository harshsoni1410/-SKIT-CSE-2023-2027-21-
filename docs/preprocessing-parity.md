# Preprocessing parity — browser vs Python

The model is trained on frames produced by Python
(`team_video_processing/preprocessing/preprocess.py`). The live web demo produces its
frames in the browser (`team_ui/frontend/src/lib/`). Both must feed the 3D CNN the
**same kind of tensor**, or the model sees a different distribution at inference time
and predictions collapse (PRD.md §7, `docs/workflow.md`).

## The contract (identical on both sides)

| Property | Value | Python | Browser |
|---|---|---|---|
| frames per sequence | **22** | `SEQ_LEN` | `SEQ_LEN` in `speechCapture.js` |
| frame size (W × H) | **112 × 80** | `FRAME_W`, `FRAME_H` | `CROP_W`, `CROP_H` in `lipDetector.js` |
| channels / order | **3, RGB** | `cv2.cvtColor(BGR→RGB)` | canvas `getImageData` is RGBA → drop A |
| value range | **0…1 float32** | `/ 255.0` | `/ 255` |
| final tensor | **(1, 22, 80, 112, 3)** | `frames_to_tensor()` | `sequenceToTensor()` |
| lip crop margin | **0.35 of box** | `CROP_MARGIN` | `CROP_MARGIN` in `lipDetector.js` |
| length fix | uniform sample / pad last | `fix_sequence_length()` | `fixSequenceLength()` |

## The one real difference — landmark source

- **Python:** dlib 68-point predictor, mouth = points 48–67.
- **Browser:** MediaPipe FaceLandmarker (468 points), mouth = the `FACEMESH_LIPS`
  vertex set (`LIP_INDICES`).

Both compute a **bounding box around the mouth landmarks**, expand it by the same
0.35 margin, crop, and resize to 112 × 80. The box is therefore equivalent even though
the landmark models differ. The inner-lip opening ratio (speaking detection) uses
dlib points 62/66 vs MediaPipe points 13/14 — again the same quantity.

## Keeping them in sync

If any value in `preprocess.py` changes, update **all** of:
`preprocess.py`, `lipDetector.js`, `speechCapture.js`, `PRD.md §7`, this file —
then rebuild the dataset and retrain.

## Checking it

- Python side: `python team_video_processing/preprocessing/preprocess.py` (self-test asserts the shape).
- Browser side: `sequenceToTensor()` returns `{ shape: [1, 22, 80, 112, 3] }`; the backend
  `coerce_tensor()` rejects anything else, so a mismatch fails loudly on the first utterance.
- Once a model is trained: `python demo/validate_model.py` must pass before trusting live predictions.
