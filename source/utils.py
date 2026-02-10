from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence, List, Tuple, TypedDict, Any
from . import custom_specialization
import tensorflow as tf
from tensorflow import keras
import yaml
import glob
import numpy as np

def cross_entropy(arrays: Sequence[np.ndarray],log_base: int | None = None) -> float:
        """
        Compute  - sum_{k,l} sum_{i=1}^m a_{i,k} * log( a_{i,l} )
        
        arrays are assumed to be raw predictions of each classificer in committee (probability predictions).
        Parameters
        ----------
        arrays : Sequence of length K. Each element is (N, C) probs, rows sum to ~1.
        K models, N samples, C classes

        Returns
        -------
        np.ndarray
        Shape (N,), intended to contain per-sample committee total entropy.
        

        -----
        """
        K = len(arrays)
        if K == 0:
            raise ValueError("arrays must contain at least one prediction array.")
        # Stack: (K, N, C)
        # P[k, i] = raw pred of model k at ith sample, of C length
        P = np.stack([np.asarray(a, dtype=np.float64) for a in arrays], axis=0)
        
        if P.ndim != 3:
            raise ValueError("Each array must have shape (N, C); stacked tensor must be (K, N, C).")
        _, N, C = P.shape
        if C < 2:
            raise ValueError("Number of classes C must be >= 2.")
        if log_base is None:
            log_base = C
        if not isinstance(log_base, int) or log_base < 2:
            raise ValueError("log_base must be an int >= 2.")
        logP = np.log(P + 1e-12) / np.log(log_base)  # (K, N, C)
        # For each sample i: build KxK matrix M_i where M_i[k,l] = -Σ_c P[k,i,c]*logP[l,i,c]
        # average over all pairs (k,l)
        # At each sample (along axis 0) take average of other (axis 1 and 2), which is 1/K^2 \sum_{k,l \in [K]}
        # It's equivalent to compute -\sum_c (1/K \sum_k P[k,i,c]) * (1/K \sum_l logP[l,i,c])
        p_bar = P.mean(axis=0)
        log_bar = logP.mean(axis=0)
        # That is to compute -\sum_c p_bar * log_bar
        
        return -(p_bar * log_bar).sum(axis=1)


def clear_folder(folder: str | Path, *, keep_folder: bool = True, dry_run: bool = False) -> None:
    
    folder = Path(folder).expanduser().resolve()

    if not folder.exists():
        print(f"[skip] not found: {folder}")
        return
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    targets = list(folder.iterdir())
    if not targets:
        print(f"[ok] empty: {folder}")
        return

    for p in targets:
        if dry_run:
            print(f"[dry-run] would delete: {p}")
            continue

        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        except Exception as e:
            print(f"[warn] failed to delete {p}: {e}")

    if not keep_folder:
        if dry_run:
            print(f"[dry-run] would delete folder: {folder}")
        else:
            try:
                folder.rmdir()
            except Exception as e:
                print(f"[warn] failed to remove folder {folder}: {e}")

    print(f"[done] cleared: {folder}")

def make_train_chunks(x: np.ndarray, y: np.ndarray, chunk_size: int = 10000):
    out = []
    n = x.shape[0]
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        name = f"train_all_{s:04d}_{e:04d}"
        out.append((name, x[s:e], y[s:e]))
    return out

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def collect_model_paths(globs: List[str]) -> List[str]:
    paths = []
    for g in globs:
        paths.extend(sorted(glob.glob(g)))
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def load_prob_models(paths: List[str]) -> List[keras.Model]:
    models = []
    for p in paths:
        m = keras.models.load_model(p)
        models.append(m)
    return models

TrainData = Tuple[np.ndarray, np.ndarray]
ValData = Tuple[np.ndarray, np.ndarray]

class HistoryDict(TypedDict, total=False):
    history: Dict[str, List[float]]
    params: Dict[str, Any]
    epoch: List[int]


class SpecializeResult(TypedDict):
    model_path: str   
    history: HistoryDict  

        
def specialize(
    new_train : TrainData,
    new_val : ValData,
    original_model : tf.keras.Model,
    new_model_path : str = 'specialized_model.keras',
    path : Path = Path('./data/specialized_models/'),
    verbose = 1,
    name : str = ''
) -> dict:
    """
    Returns (baseline_adv_model, tuned_baseline_adv_model, tuned_history_dict_or_None)
    """
    
    

    x_tr, y_tr = new_train
    x_v, y_v = new_val
    if y_tr.ndim == 1 or y_tr.shape[1] != 10:
        y_tr = keras.utils.to_categorical(y_tr, 10)
    if y_v.ndim == 1 or y_v.shape[1] != 10:
        y_v = keras.utils.to_categorical(y_v, 10)
    if not new_model_path.lower().endswith('.keras'):
        new_model_path += '.keras'
        
    model_path  = path /  Path(new_model_path)    

    if model_path.exists():
        specialized_model = keras.models.load_model(model_path)
        print(f'{model_path} already exists.')
        hist = custom_specialization.load_history(model_path) 
        
        if hist is None:
            print(f"[WARN] No history found for {model_path}. History is empty dictionary.")
            hist = {}
    else:
        print(f'{model_path} training...')
        specialized_model,hist = custom_specialization.turn_specialist(original_model, path = model_path,
                                                x_tr=x_tr, y_tr=y_tr,
                                                  x_v=x_v,   y_v=y_v,
                                                  epochs=21, learning_rate=1e-3, batch_size=128, verbose=verbose, name=f"tuned_once{name}")
        hist = {"history": hist.history, "params": hist.params, "epoch": hist.epoch}
        
    return {
    "model_path": str(model_path),
    "history": hist,
}, keras.models.load_model(model_path)