"""
Week 3 - LipSense 3D CNN architecture.

Harsh Soni, sprint task: "AI model design - designed AI model architecture for lip reading."

The model takes a fixed-length sequence of lip-region frames and predicts one word.
Input shape is locked by the shared preprocessing contract
(team_video_processing/preprocessing/preprocess.py, PRD.md section 7):

    (SEQ_LEN, FRAME_H, FRAME_W, CHANNELS) = (22, 80, 112, 3)   float32, 0..1, RGB

Design (see docs/workflow.md):

    Conv3D -> BN -> Pool3D   x3      spatial + temporal features
    GlobalAveragePooling3D          instead of Flatten -> keeps the model small
    Dense -> Dropout                classifier head
    Dense(softmax)                  one probability per word

Why GlobalAveragePooling3D and not Flatten: Flatten after the conv stack would give a
huge feature vector and a Dense layer with tens of millions of parameters (model > 100 MB,
over the PRD limit and prone to overfitting on a small self-collected dataset). Global
average pooling collapses each feature map to one number, so the head stays tiny.
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


def build_model(num_classes: int,
                input_shape: tuple[int, int, int, int] = INPUT_SHAPE,
                dropout: float = 0.5) -> tf.keras.Model:
    """Build (not compile) the 3D CNN. `num_classes` comes from the dataset folder count."""
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    model = models.Sequential(name="lipsense_3dcnn")
    model.add(layers.Input(shape=input_shape, name="lip_sequence"))

    # block 1 - keep time resolution, halve H/W
    model.add(layers.Conv3D(32, (3, 3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2)))

    # block 2 - start pooling over time
    model.add(layers.Conv3D(64, (3, 3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    # block 3
    model.add(layers.Conv3D(128, (3, 3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    # classifier head
    model.add(layers.GlobalAveragePooling3D())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_classes, activation="softmax", name="word"))

    return model


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    """Compile for single-label word classification (one-hot targets)."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# quick self-test: shapes + parameter count
if __name__ == "__main__":
    import numpy as np

    NUM_CLASSES = 13  # draft vocabulary size (PRD.md section 4)
    net = compile_model(build_model(NUM_CLASSES))
    net.summary()

    dummy = np.zeros((2, *INPUT_SHAPE), dtype="float32")
    out = net.predict(dummy, verbose=0)
    print("output shape:", out.shape, "(expected (2, %d))" % NUM_CLASSES)
    print("rows sum to 1:", np.allclose(out.sum(axis=1), 1.0))
    print("total params:", f"{net.count_params():,}")
