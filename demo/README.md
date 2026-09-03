# demo — Real-time prediction (integration)

This is where all three members' work comes together.

| File | Purpose |
|---|---|
| `predict_live.py` | Webcam → face/landmarks → lip frames → shared preprocess → 3D CNN → word + confidence on screen (no browser) |
| `validate_model.py` | Test the model on known `.npy` samples — accuracy, confusion matrix, class-wise. Must pass before trusting live predictions. |

## Run

Run from the **repo root** (the scripts add it to `sys.path` and resolve model paths from there):

```
python demo/predict_live.py            # live webcam demo
python demo/predict_live.py --camera 1 --flip

python demo/validate_model.py                    # accuracy + confusion matrix
python demo/validate_model.py --min-accuracy 0.6 # exit 1 if below (for CI gating)
```

`predict_live.py` keys: `c` calibrate (mouth closed), `q` quit.

Both work before a model is trained — `predict_live.py` shows the capture pipeline and
prints "no model"; `validate_model.py` exits with a "train first" message.

## Rule

Predict only when a full 22-frame sequence is ready. Incomplete sequence → no prediction.
Confidence `< 0.6` → "Prediction uncertain" (shown as `word?`).

## Shared code

Nothing is reimplemented here:
- face + landmarks → `team_video_processing/face_detection/detector.py`
- lip crop + normalisation → `team_video_processing/preprocessing/preprocess.py`
- model → word → `team_ai_model/training/predict.py`
