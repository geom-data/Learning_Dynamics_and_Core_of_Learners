import math
import keras, json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from keras import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model
from keras import layers
from keras.layers import Dense

"""
Utilities to specialize an existing Keras classifier by replacing its final
classification head and fine-tuning it on a (possibly narrower) task.

This module exposes:

- `lr_schedule(epoch)`: a simple stepwise learning-rate schedule used during
  training via Keras's `LearningRateScheduler` callback.
- `turn_specialist(model, path, ...)`: builds a new model that reuses the
  penultimate features of `model` (i.e., `model.layers[-2]`), attaches a new
  Dense(+Softmax) head (default: 10 classes), and fine-tunes on the provided
  train/validation sets while saving the best checkpoint (by `val_accuracy`)
  to `path`.

Assumptions
-----------
- `model.layers[-2]` is the feature layer you want to reuse. The function
  constructs `base = Model(inputs=model.inputs, outputs=model.layers[-2].output)`.
- The new head defaults to `Dense(10) -> Softmax`; adjust the Dense units to
  match your number of classes.
- Inputs `x_tr`, `x_v` are image tensors shaped like the original `model`
  input (e.g., `(N, H, W, C)`), and `y_tr`, `y_v` are one-hot encoded with a
  number of columns equal to the head's class count.
- By default, all layers remain trainable (i.e., the base is fine-tuned).
  If you want to freeze the reused backbone, set `base.trainable = False`
  before compiling.

Example
-------
>>> specialist, hist = turn_specialist(
...     model=base_model,
...     path="specialist_best.keras",
...     x_tr=x_train, y_tr=y_train,
...     x_v=x_val,   y_v=y_val,
...     epochs=21, learning_rate=1e-3, batch_size=128, verbose=1, name="_cifar10",
... )
>>> # The best weights are saved to disk; to use them explicitly:
>>> best = keras.models.load_model("specialist_best.keras")
"""


def lr_schedule(epoch):
    """
    Learning-rate schedule used during fine-tuning.

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
#     print('Learning rate: ', lr)
    return lr
def _hist_path(model_path: Path) -> Path:
    # store next to the model: foo.keras → foo.history.json
    return model_path.with_suffix(".history.json")

def _save_history(hist, model_path: Path):
    """Persist Keras History as JSON next to the model file (NumPy-safe)."""
    hpath = _hist_path(Path(model_path))

    # NumPy → JSON-safe converter
    def _to_jsonable(o):
        import numpy as _np
        if isinstance(o, _np.generic):   # e.g., np.float32, np.int64
            return o.item()
        if isinstance(o, _np.ndarray):   # arrays
            return o.tolist()
        if isinstance(o, (list, tuple)):
            return [_to_jsonable(x) for x in o]
        if isinstance(o, dict):
            return {k: _to_jsonable(v) for k, v in o.items()}
        return o

    payload = {
        "history": _to_jsonable(hist.history),
        "params":  _to_jsonable(hist.params),
        "epoch":   _to_jsonable(hist.epoch),
    }
    hpath.parent.mkdir(parents=True, exist_ok=True)
    hpath.write_text(json.dumps(payload, indent=2))
    return hpath

def load_history(model_path: Path):
    """Load History JSON if present; else return None."""
    hpath = _hist_path(Path(model_path))
    if hpath.exists():
        return json.loads(hpath.read_text())
    return None

def plot_history_curves(history_dict: dict, out_png: Path, title: str = ""):
    """Plot loss/accuracy and save to disk."""
    h = history_dict.get("history", history_dict)  # allow raw hist.history too
    fig, ax = plt.subplots(figsize=(6,4))
    if "loss" in h:       ax.plot(h["loss"], label="loss")
    if "val_loss" in h:   ax.plot(h["val_loss"], label="validation loss")
    if "accuracy" in h:   ax.plot(h["accuracy"], label="training accuracy")
    if "val_accuracy" in h: ax.plot(h["val_accuracy"], label="validation accuracy")
    ax.set_xlabel("epoch"); ax.set_ylabel("value")
    ax.set_title(title or "Training history")
    ax.legend(); fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150); plt.close(fig)
    
    
def turn_specialist(model : Model, path : str,
        x_tr: np.ndarray | None = None,
        y_tr: np.ndarray | None = None,
        x_v: np.ndarray | None = None,
        y_v: np.ndarray | None = None,
        epochs: int = 21,
        learning_rate : float = 1e-3,
        batch_size: int = 128,
        # save_each: bool = False,
        # save_bests: int | None = None,
        verbose: int = 1,
        name : str = '',
        early_stop : bool = True,
        save_history : bool = True,
    ):
        """
        Build and fine-tune a “specialist” classifier by reusing the penultimate
        features of an existing Keras `model` and attaching a fresh classification
        head.

        The function constructs a feature extractor
        `base = Model(inputs=model.inputs, outputs=model.layers[-2].output)`,
        then adds a new `Dense(10)` + `Softmax` head (change 10 to your class
        count), compiles with Adam + categorical cross-entropy, and trains on the
        provided train/validation splits. The best model (by `val_accuracy`) is
        saved to `path` via `ModelCheckpoint`.

        Note
        ----
        The returned in-memory `specialist` contains the weights from the final
        epoch, not necessarily the best checkpoint. To ensure you use the best
        validation weights, reload them from `path` with `keras.models.load_model`.

        Parameters
        ----------
        model : keras.Model
            Source model whose penultimate layer (`layers[-2]`) serves as the
            feature backbone.
        path : str
            Destination filepath for `ModelCheckpoint(save_best_only=True)`.
        x_tr, y_tr : np.ndarray, optional
            Training inputs and one-hot labels. `x_tr.shape[1:]` must be compatible
            with `model.inputs[0].shape[1:]`. `y_tr.shape[1]` must equal the number
            of classes in the new head (default 10).
        x_v, y_v : np.ndarray, optional
            Validation inputs and one-hot labels used for checkpoint selection and
            metrics reporting.
        epochs : int, default=21
            Number of fine-tuning epochs.
        learning_rate : float, default=1e-3
            Initial learning rate for Adam; further adjusted by `lr_schedule` and
            `ReduceLROnPlateau`.
        batch_size : int, default=128
            Mini-batch size.
        verbose : int, default=1
            Verbosity for Keras training and callbacks.
        name : str, default=""
            Suffix appended to newly created layer names to avoid collisions when
            creating multiple specialists from the same base.

        Returns
        -------
        specialist : keras.Model
            The compiled specialist model with the newly attached head (weights as
            of the final epoch). Best-performing weights are saved to `path`.
        hist : keras.callbacks.History
            Keras training history object.

        Raises
        ------
        ValueError
            If required data arrays are missing or shapes are incompatible with the
            base model's expected input/output.
        """
        # build specialist network
        base = Model(inputs = model.inputs, outputs = model.layers[-2].output, name=f"base{name}")
        x    = keras.Input(shape=base.input_shape[1:], name=f"in{name}")
        y    = Dense(10, name=f"dense{name}")(base(x)) # 10 can be changed to len(newtarget)
        z    = layers.Softmax(name=f"softmax{name}")(y)
        specialist = Model(inputs = x, outputs = z)
        specialist.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        



        # callbacks
        callbacks = [ModelCheckpoint(path, monitor="val_accuracy",
                                    save_best_only=True, verbose=verbose)]
        
        callbacks += [LearningRateScheduler(lr_schedule),
                        ReduceLROnPlateau(factor=np.sqrt(0.1), patience=5, min_lr=5e-7)]
        
            
        
        # fit 
        hist = specialist.fit(x_tr, y_tr, batch_size=batch_size,
                        validation_data=(x_v, y_v),
                        epochs=epochs, callbacks=callbacks, verbose=verbose)

        # save history
        if save_history:
            
            _save_history(hist, Path(path))
        else:
            pass

        # ---------- summary ----------
        metric = "val_accuracy"
        best = np.max(hist.history[metric])
        first = hist.history[metric][0]
        print(f"best {metric} {best:.3f} (first {first:.3f})")
        return specialist, hist