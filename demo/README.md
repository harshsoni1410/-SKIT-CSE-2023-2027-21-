# demo — Real-time prediction (integration)

This is where all three members' work comes together.

| File | Purpose |
|---|---|
| `predict_live.py` | Webcam -> face/landmarks -> lip frames -> shared preprocess -> 3D CNN -> word + confidence on screen |
| `validate_model.py` | Test the model on known `.npy` samples — accuracy, confusion matrix, class-wise. Must pass before connecting the frontend. |

## Run (later)

```
cd demo
python predict_live.py
```

Watch relative paths — `predict_live.py` refers to `../team_ai_model/model/model_weights.h5`,
so run the script from inside the `demo/` folder.

## Rule

Predict only when a full N-frame sequence is ready. Incomplete sequence -> no prediction.
Low confidence -> "Prediction uncertain".
