# team_video_processing — Video Processing & Data Preparation

**Owner:** Dhruv Sharma · **Sprint:** Video Processing & Data Preparation

| Path | Contents |
|---|---|
| `data_collection/` | `collect.py` — record lip-region frame sequences from the webcam |
| `preprocessing/` | `preprocess.py` — the shared preprocessing function (everyone imports this) |
| `dataset/` | word-wise data: `dataset/<word>/<sample_id>.npy` |

## Dataset structure

```
dataset/
  hello/
    000.npy      # shape (22, 80, 112, 3), float32, 0..1
    001.npy
  dog/
    000.npy
  cat/
    000.npy
```

Folder name = class label. Class order = `sorted(os.listdir("dataset"))`.
Training and demo both use this order — never hardcoded.

## Responsibilities

1. `collect.py` — face detect -> landmarks -> lip ROI -> speaking detection -> N frames -> save `.npy`.
   Show states on screen: `NOT TALKING`, `RECORDING WORD`, `NOT RECORDING`, `COLLECTED: X`.
2. `preprocess.py` — `frames_to_tensor(frames) -> np.ndarray (1, 22, 80, 112, 3)`.
   BGR->RGB, resize, /255.0, float32. Data collection, training and demo all import it.
3. Equal number of samples per word (balanced dataset).
4. Optional augmentation: horizontal flip, brightness jitter, small rotation.

## Rules

- Changing frame count / size / normalization needs team + [PRD.md § 7](../PRD.md) first.
- `dataset/` is git-ignored (large `.npy` files) — share via a drive/link.
