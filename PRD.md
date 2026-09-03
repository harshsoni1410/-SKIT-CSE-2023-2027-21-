# LipSense — Product Requirements Document (PRD)

> Team Lead (Harsh Soni), first sprint task: *"Project planning & requirements — finalize
> project requirements and overall workflow."* This document is that deliverable.

---

## 1. Problem statement

Audio-based speech recognition fails for deaf/mute users and in noisy environments. LipSense
predicts a spoken word only from **video (lip movement)** — no audio is used.

## 2. Scope (what it does)

- **Word-level** recognition — one word from a fixed vocabulary.
- **Single speaker**, **frontal face**, normal lighting.
- **Webcam** input, **real-time** (or near real-time) prediction.
- **Self-collected dataset** (recorded from our own webcams).
- Local / on-device inference. Webcam data is not uploaded to any external server.

## 3. Out of scope (state clearly in viva)

- Full-sentence / continuous speech transcription.
- Arbitrary words — only words the model was trained on.
- Multiple simultaneous speakers, side profile, very low light.
- Any language support — English words only for now.

## 4. Vocabulary (draft — locked only after training)

```
a, bye, can, cat, demo, dog, hello, here, is, lips, my, read, you
```

> This is only a draft. The **actual class list and order come from the dataset folder names
> plus the training code**. The label mapping is not finalized until the dataset is ready.
> Training and prediction must use the **exact same class ordering**.

## 5. Functional requirements

| ID | Requirement |
|---|---|
| FR-1 | Capture live video from webcam |
| FR-2 | Detect a face in each frame |
| FR-3 | Detect 68-point facial landmarks (dlib) |
| FR-4 | Crop the lip/mouth region from the landmarks |
| FR-5 | Detect "speaking / not speaking" from lip-distance threshold |
| FR-6 | Start recording frames when speaking begins, stop when it ends |
| FR-7 | Build a fixed-length sequence of frames (e.g. 22) |
| FR-8 | Preprocess frames to the model input shape |
| FR-9 | Predict word + confidence with the 3D CNN |
| FR-10 | Show "Prediction uncertain" when confidence is low |
| FR-11 | UI: webcam preview, speaking status, predicted word, confidence, history |
| FR-12 | Controls: start/stop camera, clear history |

## 6. Non-functional requirements

- Prediction latency after a sequence completes: < 1 s (target).
- Model size: < 100 MB.
- All code must be explainable in the viva — no black-box hacks.
- Must run on Windows (dev machine is Windows 11).

## 7. Model input contract (most important — everyone follows this)

Values are locked after training. Draft:

| Param | Draft value | Note |
|---|---|---|
| Frames per sequence | 22 | training == prediction |
| Frame height | 80 | |
| Frame width | 112 | |
| Channels | 3 (RGB) | or 1 (grayscale) — decided by training |
| Color format | RGB | convert from OpenCV BGR |
| Normalization | `pixel / 255.0` | training == prediction |
| dtype | float32 | |
| Final tensor | `(1, 22, 80, 112, 3)` | with batch dimension |

> **Rule:** one function in `team_video_processing/preprocessing/` produces this shape.
> Data collection, training and live demo all import that same function.

## 8. Success criteria

- Per-class accuracy report + confusion matrix on a test set.
- Correct live prediction for at least 6 words (`hello, dog, cat, you, my, bye`).
- End-to-end demo working through the frontend.

## 9. Development environment (Windows)

- **Python 3.10** (newer versions have dlib / tensorflow issues).
- `venv` virtual environment, dependencies from `requirements.txt`.
- Node.js LTS for the frontend (later).

## 10. Risks

| Risk | Mitigation |
|---|---|
| `dlib` install fails on Windows | prebuilt wheel, else `mediapipe` face mesh fallback |
| Small / imbalanced dataset -> model predicts only one word | equal samples per word, augmentation, class weights |
| TensorFlow + NumPy version clash | pinned versions in `requirements.txt` |
| Low C: disk space (npm install) | move npm cache to D:/E: when building the frontend |
| Real-time inference too slow | frame skipping, small model, predict only on a full sequence |

## 11. Future improvements (for viva)

- Sentence-level (CTC / seq2seq) lip reading.
- More speakers, side-profile robustness.
- Transfer learning from a larger public dataset (GRID, LRW).
- Mobile / edge deployment.
