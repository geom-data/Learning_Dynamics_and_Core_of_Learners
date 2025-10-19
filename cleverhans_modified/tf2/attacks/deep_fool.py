"""
DeepFool attack (TF2 / Keras refactor)
--------------------------------------
This module refactors CleverHans `deep_fool.py` to a
TensorFlow 2.x / Keras native implementation (no sessions/graphs).

It provides:
- `deepfool` function: batched eager implementation (L2 by default).
- `DeepFool` class: thin wrapper offering a `generate(x, **kwargs)`
   interface similar to the old attack, but without inheriting from the
   TF1-era `Attack` base class.

The algorithm follows Moosavi-Dezfooli et al., CVPR 2016.
https://arxiv.org/pdf/1511.04599
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal, Callable, Dict

import numpy as np
import tensorflow as tf

TensorLike = Union[np.ndarray, tf.Tensor]


@dataclass
class DeepFoolConfig:
    """
        :param max_iter: Maximum number of iteration for deepfool
        :param nb_candidate: The number of classes to test against, i.e.,
                             deepfool only consider nb_candidate classes when
                             attacking(thus accelerate speed). The nb_candidate
                             classes are chosen according to the prediction
                             confidence during implementation.
        :param overshoot: A termination criterion to prevent vanishing updates
        :param norm: The norm to use for the attack (L2, L1, or Linf)
        :param clip_min: Minimum component value for clipping
        :param clip_max: Maximum component value for clipping
        :param targeted: Whether the attack is targeted (True) or untargeted (False)
    """
    max_iter: int = 50
    nb_candidate: int = 10
    overshoot: float = 0.02
    norm: Literal["l2", "l1", "linf"] = "l2"
    clip_min: float = 0.0
    clip_max: float = 1.0
    dtype: tf.dtypes.DType = tf.float32
    targeted: bool = False
    return_details: bool = False
    

def _to_tensor(x: TensorLike, dtype: tf.dtypes.DType) -> tf.Tensor:
    return tf.convert_to_tensor(x, dtype=dtype) if not isinstance(x, tf.Tensor) else tf.cast(x, dtype)


def _predict_logits(model: Callable[[tf.Tensor], tf.Tensor], x: tf.Tensor) -> tf.Tensor:
    """Return logits; converts probabilities to log-probs if needed."""
    y = model(x, training=False)
    y = tf.convert_to_tensor(y)
    # Heuristic: detect softmax probs and convert to log
    # deepfool uses differences 
    # log of p_i = z_i - logsumexp(z)
    # log p_i - log p_j = z_i - z_j, which is difference of logits.
    if y.dtype.is_floating and y.shape.rank is not None and y.shape.rank >= 2:
        if tf.reduce_all(y >= 0.0) and tf.reduce_all(y <= 1.0):
            s = tf.reduce_sum(y, axis=-1, keepdims=True)
            if tf.reduce_all(tf.abs(s - 1.0) < 1e-3):
                return tf.math.log(tf.clip_by_value(y, 1e-8, 1.0))
    return y


def _class_grads(model: Callable[[tf.Tensor], tf.Tensor], x: tf.Tensor, cls_idx: tf.Tensor) -> tf.Tensor:
    """
    Compute gradients of selected class logits wrt x for a single sample.
    Args:
        x: [H,W,C]
        cls_idx: [K] int32 class indices
    Returns:
        grads: [K,H,W,C]
    """
    grads = []
    for j in tf.unstack(cls_idx): 
        with tf.GradientTape() as tape:
            tape.watch(x)
            # x[None, ...] is same as tf.expand_dims(x, axis=0). Batched tensor.
            logits = _predict_logits(model, x[None, ...])[0]  # [C]
            logit_j = tf.gather(logits, j)
        g = tape.gradient(logit_j, x)  # [H,W,C]
        grads.append(g)
    return tf.stack(grads, axis=0)


def _closest_step_L2(w_stack: tf.Tensor, f_stack: tf.Tensor) -> tf.Tensor:
    """
    Given w_j = ∇f_j - ∇f_y and f_j - f_y, return the minimal L2 step r.
    Args:
        w_stack: [K-1,H,W,C]
        f_stack: [K-1]
    Returns:
        r: [H,W,C]
    """
    w_flat = tf.reshape(w_stack, [tf.shape(w_stack)[0], -1])  # [K-1,D]
    w_norm_sq = tf.reduce_sum(tf.square(w_flat), axis=-1) + 1e-12  # [K-1]
    alpha = tf.abs(f_stack) / w_norm_sq                             # [K-1]
    # r_j = alpha_j * w_j ; pick minimal ||r_j||_2
    r_cands = tf.reshape(alpha[:, None] * w_flat, [tf.shape(w_stack)[0], -1])
    r_norms = tf.sqrt(tf.reduce_sum(tf.square(r_cands), axis=-1))
    j = tf.argmin(r_norms)
    r = alpha[j] * w_stack[j]
    return r


def deepfool(
    model: Callable[[tf.Tensor], tf.Tensor],
    x: TensorLike,
    max_iter: int = 50,
    nb_candidate: int = 10,
    overshoot: float = 0.02,
    norm: Literal["l2", "linf", "l1"] = "l2",
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    targeted: bool = False,
    return_details: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """
    DeepFool attack (batched).

    Args mirror the original CleverHans signature where possible.
    The model should be a tf.keras.Model (or callable) returning logits or probs.
    """
    cfg = DeepFoolConfig(
        max_iter=max_iter,
        nb_candidate=nb_candidate,
        overshoot=overshoot,
        norm=norm,
        clip_min=clip_min,
        clip_max=clip_max,
        targeted=targeted,
        return_details=return_details,
    )
    x0 = _to_tensor(x, cfg.dtype)
    if x0.shape.rank != 4:
        raise ValueError("Expected x of shape [N, H, W, C].")
    N = x0.shape[0]

    x_adv_list = []
    iters_list = []
    succ_list = []
    final_labels = []

    for i in range(N):
        xi = tf.identity(x0[i])
        xi = tf.clip_by_value(xi, cfg.clip_min, cfg.clip_max)

        logits = _predict_logits(model, xi[None, ...])[0]
        y0 = tf.argmax(logits, axis=-1, output_type=tf.int32)

        r_tot = tf.zeros_like(xi)
        changed = False
        steps = 0

        for steps in range(1, cfg.max_iter + 1):
            # Evaluate at current point
            logits = _predict_logits(model, (xi + (1.0 + cfg.overshoot) * r_tot)[None, ...])[0]
            # until class changes
            if tf.argmax(logits, axis=-1, output_type=tf.int32) != y0:
                changed = True
                break

            # Select candidate classes
            K = tf.minimum(cfg.nb_candidate, tf.shape(logits)[0])
            topk = tf.math.top_k(logits, k=K).indices
            # Ensure current class included
            if not tf.reduce_any(topk == y0):
                topk = tf.sort(tf.concat([topk, tf.reshape(tf.cast(y0, tf.int32), [1])], axis=0))

            grads = _class_grads(model, xi + (1.0 + cfg.overshoot) * r_tot, topk)  # [K,H,W,C]
            sel = tf.gather(logits, topk)                                          # [K]

            # Reference index for current class
            y_idx = tf.where(topk == tf.cast(y0, tf.int32))[0, 0]
            grad_y = grads[y_idx]
            logit_y = sel[y_idx]

            # Differences for all other classes
            mask = tf.ones_like(sel, dtype=tf.bool)
            mask = tf.tensor_scatter_nd_update(mask, [[y_idx]], [False])
            other_idx = tf.cast(tf.where(mask)[:, 0], tf.int32)

            w_stack = []
            f_stack = []
            for j in tf.unstack(other_idx):
                w_stack.append(grads[j] - grad_y)
                f_stack.append(sel[j] - logit_y)
            w_stack = tf.stack(w_stack, axis=0)  # [K-1,H,W,C]
            f_stack = tf.stack(f_stack, axis=0)  # [K-1]

            if cfg.norm != "l2":
                # For simplicity, we compute the L2 direction (original DeepFool).
                print(f"Warning: norm={cfg.norm} not implemented yet, using L2 direction.")

            r = _closest_step_L2(w_stack, f_stack)
            r_tot = r_tot + r

        x_adv = tf.clip_by_value(xi + (1.0 + cfg.overshoot) * r_tot, cfg.clip_min, cfg.clip_max)
        x_adv_list.append(x_adv)
        iters_list.append(steps)
        succ_list.append(1 if changed else 0)
        final_labels.append(int(tf.argmax(_predict_logits(model, x_adv[None, ...])[0])))

    x_adv_tf = tf.stack(x_adv_list, axis=0)
    x_adv_np = x_adv_tf.numpy()
    if not return_details:
        return x_adv_np
    details = dict(
        iters=np.array(iters_list, dtype=np.int32),
        success=np.array(succ_list, dtype=np.int32),
        final_labels=np.array(final_labels, dtype=np.int32),
    )
    return x_adv_np, details


class DeepFool:
    """
    A light wrapper providing a `generate(x, **kwargs)` method for legacy-style use.
    """
    def __init__(self, model, **defaults):
        self.model = model
        self.defaults = defaults

    def generate(self, x, **kwargs):
        params = {**self.defaults, **kwargs}
        return deepfool(self.model, x, **params)


__all__ = ["DeepFool", "DeepFoolConfig", "deepfool"]


# We may update the implementation to support Linf/L1 norms in the future.
# See below

'''
def _closest_step(w_stack: tf.Tensor, f_stack: tf.Tensor, norm: str = "l2") -> tf.Tensor:
    """
    Return minimal-norm step r to the linearized boundary under given norm.
    w_stack: [K-1,H,W,C] where w_j = ∇f_j - ∇f_y
    f_stack: [K-1]       where f_j - f_y (scalars)
    norm: "l2" | "linf" | "l1"
    """
    eps = 1e-12
    Km1 = tf.shape(w_stack)[0]
    flat = tf.reshape(w_stack, [Km1, -1])             # [K-1, D]
    abs_flat = tf.abs(flat)

    if norm == "l2":
        denom = tf.reduce_sum(tf.square(flat), axis=1) + eps          # ||w||_2^2
        alpha = tf.abs(f_stack) / denom                                # [K-1]
        r_flat = alpha[:, None] * flat                                 # alpha * w
        r_norms = tf.sqrt(tf.reduce_sum(tf.square(r_flat), axis=1))    # == |f|/||w||_2

    elif norm == "linf":
        denom = tf.reduce_sum(abs_flat, axis=1) + eps                  # ||w||_1
        alpha = tf.abs(f_stack) / denom
        r_flat = alpha[:, None] * tf.sign(flat)                        # alpha * sign(w)
        r_norms = tf.reduce_max(tf.abs(r_flat), axis=1)                # == alpha

    elif norm == "l1":
        # place all mass on argmax coordinate
        kmax = tf.argmax(abs_flat, axis=1, output_type=tf.int32)       # [K-1]
        w_max = tf.gather_nd(abs_flat, tf.stack([tf.range(Km1), kmax], axis=1)) + eps
        alpha = tf.abs(f_stack) / w_max                                # [K-1]
        # build sparse r_flat with one nonzero per row
        r_flat = tf.zeros_like(flat)
        signs = tf.gather_nd(tf.sign(flat), tf.stack([tf.range(Km1), kmax], axis=1))
        updates = alpha * signs
        r_flat = tf.tensor_scatter_nd_update(r_flat,
                                             tf.stack([tf.range(Km1), kmax], axis=1),
                                             updates)
        r_norms = tf.reduce_sum(tf.abs(r_flat), axis=1)                # == alpha

    else:
        raise ValueError("norm must be 'l2', 'linf', or 'l1'")

    j = tf.argmin(r_norms)                                             # closest boundary
    r = tf.reshape(r_flat[j], tf.shape(w_stack)[1:])                   # [H,W,C]
    return r


'''