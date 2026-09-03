# team_ai_model — AI Model Development

**Owner:** Harsh Soni (Team Lead) · **Sprint:** AI Model Development

| Path | Contents |
|---|---|
| `training/` | `model.py` (3D CNN), `dataset.py` + `train.py` (data + training), `evaluate.py` (metrics), `predict.py` (`LipReader` — model → word) |
| `model/` | `model_weights.h5` (trained), `face_weights.dat` (dlib 68-point predictor) |
| `outputs/` | training history graphs, confusion matrix, per-class accuracy, logs |

## Responsibilities

1. Finalize requirements + workflow — [PRD.md](../PRD.md), [docs/workflow.md](../docs/workflow.md). *(done)*
2. Design the 3D CNN — `training/model.py`. *(done)* 3× (Conv3D → BN → Pool3D) →
   GlobalAveragePooling3D → Dense → Dropout → Softmax. Uses global average pooling
   instead of Flatten so the model stays well under the 100 MB PRD limit.
3. Training pipeline: train/val split, categorical crossentropy, Adam, accuracy metric,
   ModelCheckpoint on best val_accuracy, save training history.
4. Evaluation: confusion matrix + per-class accuracy saved in `outputs/`.
5. Produce `class_names.json` from the dataset folder order so demo and training share one mapping.

## Rules

- Do not write preprocessing here — import from `team_video_processing/preprocessing/`.
- Whatever input shape is trained, update it in [PRD.md § 7](../PRD.md).
- `model/face_weights.dat` = dlib's `shape_predictor_68_face_landmarks.dat` (renamed). It is ~100 MB;
  it is git-ignored — share via Git LFS or an external link.
