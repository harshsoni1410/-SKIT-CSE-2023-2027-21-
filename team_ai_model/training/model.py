"""
Week 3 - LipSense 3D CNN architecture. Week 9 - added a temporal (BiLSTM) head.

Harsh Soni, sprint task: "AI model design - designed AI model architecture for lip reading."

The model takes a fixed-length sequence of lip-region frames and predicts one word.
Input shape is locked by the shared preprocessing contract
(team_video_processing/preprocessing/preprocess.py, PRD.md section 7):

    (SEQ_LEN, FRAME_H, FRAME_W, CHANNELS) = (22, 80, 112, 3)   float32, 0..1, RGB

Two architectures, selected with `architecture=`:

    "cnn"       Conv3D -> BN -> Pool3D  x3 -> GlobalAveragePooling3D -> Dense -> Softmax
                (original Week 3 design; fast, but collapses frame order into an average
                so two words that share the same mouth shapes in a different order, or
                differ only in *when* the mouth moves, can look identical to it.)

    "cnn_lstm"  (default) same Conv3D stack but WITHOUT pooling over time, so all 22 time
                steps survive -> per-frame spatial features -> Bidirectional LSTM over
                time -> Dense -> Softmax.

Why the switch to cnn_lstm: the project's target vocabulary is now rhyming minimal pairs
(cat/bat/hat/mat/rat/sat) that differ only in the first ~2-3 frames (the initial
consonant's mouth shape) and are otherwise near-identical. GlobalAveragePooling3D
averages that early distinguishing motion together with 20 identical "-at" frames, so
the signal we need most gets diluted the most. A BiLSTM sees the frames in order and can
weight that brief opening differently from the rest of the sequence. "cnn" is kept for
comparison/ablation and for cases where speed matters more than that distinction.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

# --- input contract (must match preprocessing.preprocess) ---
SEQ_LEN = 22
FRAME_H = 80
FRAME_W = 112
CHANNELS = 3
INPUT_SHAPE = (SEQ_LEN, FRAME_H, FRAME_W, CHANNELS)


def _build_cnn(input_shape, dropout) -> tf.keras.Model:
    model = models.Sequential(name="lipsense_3dcnn")
    model.add(layers.Input(shape=input_shape, name="lip_sequence"))

    model.add(layers.Conv3D(32, (3, 3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(layers.Conv3D(64, (3, 3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(layers.Conv3D(128, (3, 3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(layers.GlobalAveragePooling3D())
    return model


def _build_cnn_lstm(input_shape, dropout) -> tf.keras.Model:
    model = models.Sequential(name="lipsense_cnn_lstm")
    model.add(layers.Input(shape=input_shape, name="lip_sequence"))

    # block 1: full time resolution, halve H/W (the onset consonant needs precise timing)
    model.add(layers.Conv3D(24, (3, 3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2)))

    # block 2: halve time too (22 -> 11 steps is still fine-grained enough to separate
    # the first 2-3 onset frames from the rest, and roughly halves compute for the rest
    # of the network + the LSTM - matters on a CPU-only laptop with no GPU)
    model.add(layers.Conv3D(48, (3, 3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(layers.Conv3D(96, (3, 3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2)))

    # (T, H, W, C) -> (T, C) per-frame feature vector, order kept intact
    model.add(layers.TimeDistributed(layers.GlobalAveragePooling2D()))
    model.add(layers.Bidirectional(layers.LSTM(64, dropout=dropout)))
    return model


def build_model(num_classes: int,
                input_shape: tuple[int, int, int, int] = INPUT_SHAPE,
                dropout: float = 0.5,
                architecture: str = "cnn_lstm") -> tf.keras.Model:
    """Build (not compile) the model. `num_classes` comes from the dataset folder count."""
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    if architecture == "cnn":
        trunk = _build_cnn(input_shape, dropout)
    elif architecture == "cnn_lstm":
        trunk = _build_cnn_lstm(input_shape, dropout)
    else:
        raise ValueError(f"unknown architecture {architecture!r}, expected 'cnn' or 'cnn_lstm'")

    trunk.add(layers.Dense(128, activation="relu"))
    trunk.add(layers.Dropout(dropout))
    trunk.add(layers.Dense(num_classes, activation="softmax", name="word"))
    trunk._name = f"lipsense_{architecture}"
    return trunk


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3,
                  label_smoothing: float = 0.0) -> tf.keras.Model:
    """Compile for single-label word classification (one-hot targets).

    label_smoothing > 0 softens the one-hot targets a little (e.g. 0.05). Useful here
    because some of our classes (bat/mat, cat/hat) are genuinely close to each other in
    viseme space, so a model pushed to be 100% confident on every training example tends
    to overfit noise instead of the real distinguishing motion.
    """
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


# quick self-test: shapes + parameter count, both architectures
if __name__ == "__main__":
    import numpy as np

    NUM_CLASSES = 6  # cat/bat/hat/mat/rat/sat (PRD.md section 4)
    dummy = np.zeros((2, *INPUT_SHAPE), dtype="float32")

    for arch in ("cnn_lstm", "cnn"):
        net = compile_model(build_model(NUM_CLASSES, architecture=arch), label_smoothing=0.05)
        out = net.predict(dummy, verbose=0)
        assert out.shape == (2, NUM_CLASSES)
        assert np.allclose(out.sum(axis=1), 1.0)
        print(f"[{arch}] output {out.shape}  params {net.count_params():,}")
