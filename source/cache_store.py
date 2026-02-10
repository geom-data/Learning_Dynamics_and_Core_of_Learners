"""
cache_store.py

Cache for model predictions and ensemble metrics used in this project.

This module stores:
- per-model prediction probabilities (NumPy .npy) for a given split, and
- per-committee entropy arrays (NumPy .npy) identified by a given committee signature.

The cache layout (under root) is fixed:

    preds/<sample_name>/<model_stem>.npy
        Raw prediction probabilities with shape (N, C) in float32. (C = 10 for CIFAR-10)

    metrics/<sample_name>/entropy__<committee_sig>.npy
        Per-sample committee entropy with shape (N,) in float64.
"""

import os, pickle, hashlib
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np

def stem_all(path: str) -> str:
    """
    Return the filename stem with all suffixes removed.

    Examples
    --------
    >>> stem_all("model.keras")
    'model'
    >>> stem_all("model.npy.keras")
    'model'
    """
    p = Path(path)
    while p.suffix:
        p = p.with_suffix("")
    return p.name

class ResultStore:
    """
    Cache store for predictions and derived metrics.

    root:
        Cache root directory. 
        Subdirectories preds/ and metrics/ will be created if missing.

    """
    def __init__(self, root: str = "./data/cache"):
        self.root = Path(root)
        (self.root / "preds").mkdir(parents=True, exist_ok=True)
        (self.root / "metrics").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _committee_sig(model_paths: List[str]) -> str:
        """
        Compute a committee signature from model stems (order-insensitive).

        Parameters
        ----------
        model_paths:
            List of model file paths.

        Returns
        -------
        str
            Short hex signature suitable for filenames, e.g., "a1b2c3d4e5f6".

        Notes
        -----
        - Signature is computed from sorted stems joined by '||' and hashed with SHA-1.
        - Truncation to 12 hex chars (~48 bits) is plenty for this project scale.
        """
        ids = sorted(stem_all(p) for p in model_paths)
        joined = "||".join(ids)
        return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _stem(path: str, _stem_all_ : bool = True) -> str:
        if _stem_all_:
            return stem_all(path)
        else:
            return Path(path).stem
    @staticmethod
    def int_from_model_path(model_path: str) -> int:
        """
        Build a unique integer ID from the filename (no extensions).
        Example file path : '/path/to/n_3_v1_cifar10_2.keras'
        Example stem      : 'n_3_v1_cifar10_2'"""
        stem = stem_all(model_path)
        h = hashlib.blake2s(stem.encode('utf-8'), digest_size=8).digest()
        return int.from_bytes(h, 'big')

    # raw predictions 
    def pred_path(self, model_path: str, sample_name: str) -> Path:
        """
        Return the canonical cache path for predictions.

        Parameters
        ----------
        model_path:
            Path to the saved model (used only for its stem).
        sample_name:
            Identifier of the evaluation split (e.g., "clean_test", "APGD_strong_test").

        Returns
        -------
        pathlib.Path
            Path like: root/preds/<sample_name>/<model_stem>.npy
        """
        return self.root / "preds" / sample_name / f"{stem_all(model_path)}.npy"
    
    def get_pred(self, model_path: str, sample_name: str) -> Optional[np.ndarray]:
        """
        Load cached predictions if present.

        Returns
        -------
        np.ndarray or None
            Array of shape (N, C) if cached, else None.
        """
        fp = self.pred_path(model_path, sample_name)
        return np.load(fp) if fp.exists() else None
    
    def set_pred(self, model_path: str, sample_name: str, preds: np.ndarray) -> str:
        """
        Save predictions to cache.

        Parameters
        ----------
        preds:
            Prediction probabilities of shape (N, C). Saved as float32.

        Returns
        -------
        str
            File path written.
        """
        outdir = self.root / "preds" / sample_name
        outdir.mkdir(parents=True, exist_ok=True)
        fp = outdir / f"{stem_all(model_path)}.npy"
        np.save(fp, preds.astype(np.float32))
        return str(fp)
    
    # entropy metrics
    def entropy_path(self, model_paths: List[str], sample_name: str) -> Path:
        """
        Return the canonical cache path for committee entropy.

        Returns
        -------
        pathlib.Path
            Path like: root/metrics/<sample_name>/entropy__<sig>.npy
        """
        sig = self._committee_sig(model_paths)
        return self.root / "metrics" / sample_name / f"entropy__{sig}.npy"
    
    def get_entropy(self, model_paths: List[str], sample_name: str) -> Optional[np.ndarray]:
        """
        Load cached committee entropy if present.

        Returns
        -------
        np.ndarray or None
            Array of shape (N,) if cached, else None.
        """
        fp = self.entropy_path(model_paths, sample_name)
        return np.load(fp) if fp.exists() else None

    def set_entropy(self, model_paths: List[str], sample_name: str, ent: np.ndarray) -> str:
        """
        Save committee entropy to cache.

        Parameters
        ----------
        ent:
            Per-sample entropy array of shape (N,). Saved as float64.

        Returns
        -------
        str
            File path written.
        """
        outdir = self.root / "metrics" / sample_name
        outdir.mkdir(parents=True, exist_ok=True)
        sig = self._committee_sig(model_paths)
        fp = outdir / f"entropy__{sig}.npy"
        np.save(fp, ent.astype(np.float64))
        return str(fp)
    