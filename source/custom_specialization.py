"""
custom_specialization.py

Project utility to build a CIFAR-10 "specialist" from an existing TF/Keras classifier.

What this file does
-------------------
- Reuses the penultimate layer of a given Keras model as a feature backbone.
- Replaces the final classification head with a new Dense(+Softmax) head.
- Fine-tunes the resulting specialist for a short schedule (default: 21 epochs).
- Saves the best checkpoint by validation accuracy and optionally saves the training history.

Motivation
----------
This method is motivated by the logifold viewpoint, where each classifier is treated as a chart (local observer) on a given dataset. See the paper https://www.mdpi.com/2075-1680/14/8/599

In this perspective, multiple charts may share domain, while differing in their target. This naturally leads to reuse of a common backbone with task-specific heads.

Concretely, a chart may operate on the same inputs but with different target structures, e.g.
	
 - Different label groups with the same backbone. One classifier targets {dog, cat, horse}, while another targets a coarser partition such as {{dog, cat}, horse}. These tasks share essentially the same “flattened target”, but require different output heads (3-way vs 2-way).
	
 - Experts for a subset of classes. Starting from a N-class classifier {c1, …, cN}, we may want specialists restricted to {c1, …, c5}. A practical way to obtain such experts is to keep the penultimate features and replace the final Dense(+Softmax) head.

Assumptions (project-specific)
------------------------------
- The input model is a TF/Keras `Model`.
- The penultimate layer (`model.layers[-2]`) is the feature layer to reuse.
- Labels are one-hot encoded (CIFAR-10 => 10 classes).

Notes
-----
- ModelCheckpoint monitor uses `val_accuracy` (recommended metric name in modern Keras).   (https://keras.io/api/callbacks/model_checkpoint/)
- LearningRateScheduler calls `schedule(epoch, lr)`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


def lr_schedule(epoch):
    """
    Starts at 1e-3 and decays stepwise as training progresses:
      - epochs 0-5   -> 1e-3
      - epochs 6-10  -> 1e-4  (* 1e-1)
      - epochs 11-15 -> 1e-5  (* 1e-2)
      - epochs 16-18 -> 1e-6  (* 1e-3)
      - epochs >18   -> 5e-7  (* 0.5e-3)

    Parameters
    ----------
    epoch : int
        Zero-based epoch index provided by Keras.

    Returns
    -------
    float
        Learning rate for the current epoch, to be consumed by
        `keras.callbacks.LearningRateScheduler`.
    """
    
    lr = 1e-3
    if epoch > 18:
        lr *= 0.5e-3
    elif epoch > 15:
        lr *= 1e-3
    elif epoch > 10:
        lr *= 1e-2
    elif epoch > 5:
        lr *= 1e-1
    return lr

def _hist_path(model_path: Path) -> Path:
    # store next to the model: foo.keras → foo.history.json
    return model_path.with_suffix(".history.json")

def _save_history(hist: tf.keras.callbacks.History, model_path: Path) -> Path:
    """Persist Keras History as JSON next to the model file (NumPy-safe)."""
    hpath = _hist_path(Path(model_path))

    def _to_jsonable(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (list, tuple)):
            return [_to_jsonable(x) for x in o]
        if isinstance(o, dict):
            return {k: _to_jsonable(v) for k, v in o.items()}
        return o

    payload = {
        "history": _to_jsonable(hist.history),
        "params": _to_jsonable(getattr(hist, "params", {})),
        "epoch": _to_jsonable(getattr(hist, "epoch", [])),
    }
    hpath.parent.mkdir(parents=True, exist_ok=True)
    hpath.write_text(json.dumps(payload, indent=2))
    return hpath

def load_history(model_path: Path) -> Optional[dict]:
    """Load History JSON if present; else return None."""
    hpath = _hist_path(Path(model_path))
    return json.loads(hpath.read_text()) if hpath.exists() else None


def turn_specialist(
    model: tf.keras.Model,
    path: str,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_v: np.ndarray,
    y_v: np.ndarray,
    *,
    num_classes: int = 10,
    epochs: int = 21,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    verbose: int = 1,
    name: str = "",
    early_stop: bool = False,
    save_history: bool = True,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Build and fine-tune a specialist by reusing `model.layers[-2]` as backbone.

    Parameters
    ----------
    model:
        Base classifier. We reuse `model.layers[-2].output` as features.
    path:
        Output checkpoint path (typically ends with `.keras`).
        Best checkpoint is selected by `val_accuracy`.
    x_tr, y_tr:
        Training data and one-hot labels.
    x_v, y_v:
        Validation data and one-hot labels.
    num_classes:
        Number of classes for the new head (CIFAR-10 => 10).
    epochs, learning_rate, batch_size, verbose:
        Usual Keras training hyperparameters.
    name:
        Suffix for new layer names to avoid collisions.
    early_stop:
        If True, enable EarlyStopping on `val_accuracy`.
    save_history:
        If True, save `<path>.history.json`.

    Returns
    -------
    specialist, hist
        The in-memory model is the final-epoch weights; the best weights are saved to `path`.
    """
    if x_tr is None or y_tr is None or x_v is None or y_v is None:
        raise ValueError("x_tr, y_tr, x_v, y_v must be provided.")

    # backbone from penultimate layer
    base = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output, name=f"base{name}")

    x = tf.keras.Input(shape=base.input_shape[1:], name=f"in{name}")
    logits = tf.keras.layers.Dense(num_classes, name=f"dense{name}")(base(x))
    probs = tf.keras.layers.Softmax(name=f"softmax{name}")(logits)
    specialist = tf.keras.Model(inputs=x, outputs=probs, name=f"specialist{name}")

    specialist.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=verbose,
        ),
        tf.keras.callbacks.LearningRateScheduler(lr_schedule),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=float(np.sqrt(0.1)),
            patience=5,
            min_lr=5e-7,
        ),
    ]
    

    if early_stop:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=6,
                restore_best_weights=False, 
                verbose=verbose,
            )
        )

    hist = specialist.fit(
        x_tr, y_tr,
        batch_size=batch_size,
        validation_data=(x_v, y_v),
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )

    if save_history:
        _save_history(hist, Path(path))

    metric = "val_accuracy"
    best = float(np.max(hist.history.get(metric, [np.nan])))
    first = float(hist.history.get(metric, [np.nan])[0])
    print(f"best {metric} {best:.3f} (first {first:.3f})")

    return specialist, hist
    