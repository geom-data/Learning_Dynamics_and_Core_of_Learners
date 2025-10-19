# cache_store.py
import os, json, pickle, hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

def stem_all(path: str) -> str:
    """filename with *all* suffixes removed (handles .keras.npy, .keras.keras, etc.)."""
    p = Path(path)
    while p.suffix:
        p = p.with_suffix('')
    return p.name

class ResultStore:
    """
    Filesystem-backed cache for predictions and metrics.

    Layout (under root):
      preds/<sample_name>/<model_stem>.npy                 # raw probs (N, C)
      metrics/<sample_name>/entropy__<committee_sig>.npy   # committee cross-entropy per sample (N,)
      index/preds.pkl, index/metrics.pkl                   # (optional) dict-style index mapping to file paths
      
      pred index example:
      {
        (12967323447745904301, "cifar10_original_test"): "./data/cache/preds/cifar10_original_test/resnet56v1_ff.npy",
        ...
        }
      metrics index example:
      {
        ("entropy", "3e9ac7d4b2f1", "cifar10_original_test"): "./data/cache/metrics/cifar10_original_test/entropy__3e9ac7d4b2f1.npy",
        ...
      }
      
      Here, the big integer is a stable hash-based ID derived from the model filename (without extensions).
    """
    def __init__(self, root: str = "./data/cache"):
        self.root = Path(root)
        (self.root / "preds").mkdir(parents=True, exist_ok=True)
        (self.root / "metrics").mkdir(parents=True, exist_ok=True)
        (self.root / "index").mkdir(parents=True, exist_ok=True)
        self._preds_idx_path   = self.root / "index" / "preds.pkl"
        self._metrics_idx_path = self.root / "index" / "metrics.pkl"
        self._preds_idx   = self._load_idx(self._preds_idx_path)
        self._metrics_idx = self._load_idx(self._metrics_idx_path)

    def _load_idx(self, p: Path) -> Dict:
        if p.exists():
            with open(p, "rb") as f: return pickle.load(f)
        return {}

    def _dump_idx(self, p: Path, idx: Dict):
        '''
        atomic-ish update — write to a .tmp file, then os.replace to the final path. 
        This reduces the chance of a half-written file.
        '''
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as f: pickle.dump(idx, f)
        os.replace(tmp, p)

    @staticmethod
    def _stem(path: str, _stem_all_ : bool = True) -> str:
        if _stem_all_:
            return stem_all(path)
        else:
            return Path(path).stem

    @staticmethod
    def _committee_sig(model_paths: List[str]) -> str:
        '''
        stable signature based on model *stems* without order
        
        Result is used as a filename-friendly committee signature, e.g. to name an entropy file:
            entropy__a1b2c3d4e5f6.npy.
        '''
        ids = [stem_all(p) for p in model_paths]
        ids = sorted(ids)
        joined = "||".join(ids)
        # joined.encode("utf-8") → bytes for hashing.
        # hashlib.sha1(...) → computes a SHA-1 hash of that string (not for security here, just an ID).
        # .hexdigest() → 40-char hex string; [:12] truncates to a 12-char short signature (~48 bits).
        
        # Name collisions: two different folders with the same filename produce the same stem. 
        # Use .name (keeps extension) or include the full path if that matters.
        
        # Collision risk: 12 hex chars ≈ 2^48 space—fine for small projects. 
        # If you want extra safety, use more chars (e.g., [:16]) or blake2s(digest_size=8..12).
        
        return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def int_from_model_path(model_path: str, *, split_on: Optional[str] = None) -> int:
        """
        Build a unique integer ID from the filename (no extensions).
        Example file path : '/path/to/n_3_v1_cifar10_2.keras'
        If split_on is given, split the stem on that string and return the last part as int.
        """
        stem = stem_all(model_path)
        h = hashlib.blake2s(stem.encode('utf-8'), digest_size=8).digest()
        return int.from_bytes(h, 'big')

    # ---------- raw predictions ----------
    def get_pred(self, model_path: str, sample_name: str) -> Optional[np.ndarray]:
        key = (self.int_from_model_path(model_path), sample_name)
        fp = self._preds_idx.get(key) #file path
        if fp and Path(fp).exists():
            return np.load(fp)
        # fallback to conventional path
        fp2 = self.root / "preds" / sample_name / f"{self._stem(model_path)}.npy"
        if fp2.exists():
            self._preds_idx[key] = str(fp2); self._dump_idx(self._preds_idx_path, self._preds_idx)
            return np.load(fp2)
        return None

    def set_pred(self, model_path: str, sample_name: str, preds: np.ndarray) -> str:
        outdir = self.root / "preds" / sample_name
        outdir.mkdir(parents=True, exist_ok=True)
        fp = outdir / f"{self._stem(model_path)}.npy"
        np.save(fp, preds.astype(np.float32))
        key = (self.int_from_model_path(model_path), sample_name)
        self._preds_idx[key] = str(fp); self._dump_idx(self._preds_idx_path, self._preds_idx)
        return str(fp)
    ## NOTE We don't currently use confusion matrices.
    # # ---------- confusion matrix ----------
    # def get_cm(self, model_path: str, sample_name: str) -> Optional[np.ndarray]:
    #     key = ("cm",self._stem(model_path), sample_name)
    #     fp = self._metrics_idx.get(key)
    #     if fp and Path(fp).exists():
    #         return np.load(fp)
    #     fp2 = self.root / "metrics" / sample_name / f"cm__{self._stem(model_path)}.npy"
    #     if fp2.exists():
    #         self._metrics_idx[key] = str(fp2); self._dump_idx(self._metrics_idx_path, self._metrics_idx)
    #         return np.load(fp2)
    #     return None

    # def set_cm(self, model_path: str, sample_name: str, cm: np.ndarray) -> str:
    #     outdir = self.root / "metrics" / sample_name
    #     outdir.mkdir(parents=True, exist_ok=True)
    #     fp = outdir / f"cm__{self._stem(model_path)}.npy"
    #     np.save(fp, cm.astype(np.int64))
    #     key = ("cm", self._stem(model_path), sample_name)
    #     self._metrics_idx[key] = str(fp); self._dump_idx(self._metrics_idx_path, self._metrics_idx)
    #     return str(fp)

    # ---------- committee entropy (AdvLogifold-style cross-entropy among models) ----------
    def get_entropy(self, model_paths: List[str], sample_name: str) -> Optional[np.ndarray]:
        sig = self._committee_sig(model_paths)
        key = ("entropy", sig, sample_name)
        fp = self._metrics_idx.get(key)
        if fp and Path(fp).exists():
            return np.load(fp)
        fp2 = self.root / "metrics" / sample_name / f"entropy__{sig}.npy"
        if fp2.exists():
            self._metrics_idx[key] = str(fp2); self._dump_idx(self._metrics_idx_path, self._metrics_idx)
            return np.load(fp2)
        return None

    def set_entropy(self, model_paths: List[str], sample_name: str, ent: np.ndarray) -> str:
        outdir = self.root / "metrics" / sample_name
        outdir.mkdir(parents=True, exist_ok=True)
        sig = self._committee_sig(model_paths)
        fp = outdir / f"entropy__{sig}.npy"
        np.save(fp, ent.astype(np.float64))
        key = ("entropy", sig, sample_name)
        self._metrics_idx[key] = str(fp); self._dump_idx(self._metrics_idx_path, self._metrics_idx)
        return str(fp)