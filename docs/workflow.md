# LipSense — End-to-End Workflow

This explains the full system flow. Use it to explain the project in the viva.

---

## Full pipeline

```
Webcam                     Live video frames (OpenCV: cv2.VideoCapture)
  |
  v
Face detection             dlib.get_frontal_face_detector()
  |                        -> face bounding box
  v
Facial landmarks           dlib shape_predictor (68 points)
  |                        -> exact points for eyes, nose, mouth
  v
Lip ROI extraction         Landmarks 48-67 = mouth
  |                        -> bounding box around those points, crop, resize (112x80)
  v
Speaking detection         Distance between inner lips (points 62 & 66)
  |                        distance > threshold        -> TALKING
  |                        distance < threshold (N frames) -> NOT TALKING
  v
Frame sequence             While TALKING, append lip crops to a buffer.
collection                 When the utterance ends, take exactly N frames
  |                        (pad if fewer, uniformly sample if more).
  v
Preprocessing              BGR->RGB, resize (112x80), /255.0, float32
  |                        shape: (N, 80, 112, 3) -> batch: (1, N, 80, 112, 3)
  |                        (uses the same function as training)
  v
3D CNN                     Conv3D -> Pool3D -> Conv3D -> Pool3D -> Flatten
  |                        -> Dense -> Dropout -> Softmax (num_classes)
  v
Prediction                 softmax vector -> argmax = class index
  |                        index -> word (label map)
  |                        max(softmax) = confidence
  v
UI                         "Predicted Word: HELLO   Confidence: 92%"
                           (confidence < threshold -> "Prediction uncertain")
                           + prediction history
```

---

## Why each step (viva Q&A)

### Why OpenCV?
To capture webcam frames, resize/crop/convert color, and draw on screen. Fast, standard CV library.

### Why dlib / facial landmarks? Isn't a face box enough?
A face box only says "the face is here". We need only the **lips**. The 68-point landmark model
gives points 48-67 exactly on the mouth, so we can crop a precise lip ROI even if the head moves.

### Why a 3D CNN? Why not a normal (2D) CNN?
Lip reading needs two things:
1. **Spatial** — the shape/appearance of the lips in a single frame (a 2D CNN does this).
2. **Temporal** — how the lips move across frames. "b" and "m" can look identical in one frame;
   the difference is in the **movement**.

A 2D CNN sees only one image. A **3D CNN kernel (t x h x w)** slides across several frames at
once, so it learns spatial + temporal features together. That is why 3D CNNs are used for
video/gesture tasks.

### How does speaking detection work?
We measure the vertical distance between the inner upper lip (point 62) and inner lower lip
(point 66), normalized by mouth width. This distance changes while speaking. We set a threshold:
above it = TALKING; below it for several consecutive frames = the utterance has ended.

### Why a fixed frame count (e.g. 22)?
A 3D CNN needs a fixed input shape. Each word takes a different amount of time, so we normalize
every utterance to the same N frames (uniform sampling / padding).

### How is confidence computed?
The softmax output is a probability distribution (sums to 1). The largest probability is the
confidence. If even that is low (e.g. < 0.6), the model is unsure — better to show "Prediction
uncertain" than a confident wrong word.

### Why must training and prediction preprocessing be identical?
The model learned on a specific distribution (same size, color, normalization). If live data
differs, the model gets confused and may keep predicting one class. Hence one shared
preprocessing function.

---

## Known limitations (mention in viva)

- Fixed vocabulary — will not recognise a word it was not trained on.
- Single speaker, frontal face, decent lighting required.
- Small self-collected dataset — limited accuracy.
- Homophenes (e.g. "b"/"p"/"m") look visually similar and can be confused.
