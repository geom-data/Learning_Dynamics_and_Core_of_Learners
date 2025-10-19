"""
AdvLogifold: entropy-routed predictor
- Computes per-sample cross-entropy among Judge committee member predictions
- Routes low-entropy vs high-entropy samples to two committees
- Predicts on a flat 'full-and-fine' target (root only)
#TODO We need to adapt this voting strategy with non-trivial target tree

Folder assumptions :
logifold_modules/
  adv_logifold.py
  logifoldv1_4_modified.py
"""

#TODO Check if this script works
from __future__ import annotations

from typing import Sequence, Optional, Dict, List, Tuple
import numpy as np
from pathlib import Path
import hashlib
import matplotlib.pyplot as plt

from logifold_modules.logifoldv1_4_modified import Logifold

def plot_disagreements(disagreements: np.ndarray | dict[str, np.ndarray], title: str = "Disagreements", save_path: str = None):
    '''
    disagreements is either a numpy array of entropies, or a dictionary of entropy whose keys are the name of sample.
    '''
    plt.figure(figsize=(16, 6))
    plt.boxplot(disagreements if isinstance(disagreements, np.ndarray) else list(disagreements.values()))
    plt.title(title)
    if isinstance(disagreements, dict):
        plt.xticks(ticks=range(1, len(disagreements) + 1), labels=list(disagreements.keys()))
    else:
        plt.xlabel("Sample Index")
    plt.ylabel("Disagreement")
    plt.show()
    if save_path is not None:
        plt.savefig(f"{save_path}.png", dpi=150)
def get_statistics(disagreements: np.ndarray) -> dict[str, float]:
    '''
    disagreements is a numpy array of entropies.
    return dictionary of average, median, std, 1Q, 3Q.
    '''
    return {
        "average": float(np.mean(disagreements)),
        "median": float(np.median(disagreements)),
        "std": float(np.std(disagreements)),
        "1Q": float(np.percentile(disagreements, 25)),
        "3Q": float(np.percentile(disagreements, 75)),
    }
def flatTuple(target, ascending: bool = True):
    out = ()
    for a in target:
        out += tuple(a)
    if ascending:
        out = tuple(sorted(out))
    return out


class AdvLogifold(Logifold):
    """
    Override of `predict` that:
      - computes per-instance entropy,
      - routes to one of two committees by a threshold,
      - predicts on a single flat 'full-and-fine' target (root only).
    """
    
    @staticmethod
    def _stem_all(path: str) -> str:
        p = Path(str(path))
        while p.suffix:
            p = p.with_suffix('')
        return p.name
    
    def _committee_sig_from_keys(self, keys: list) -> Optional[str]:
        """
        Reproduce the signature used by your store: sha1 over sorted stems, truncated to 12 hex chars.
        If we cannot resolve any stems, we still compute on str(k); but for an exact match to
        previously saved files, ensure that either:
          - keys were those stems originally, or
          - you provide self.model_path_map so stems match.
        """
        
        known, unknown = self.model_source_name(keys)
        if not known:
            return None
        joined = "||".join(sorted(known))
        return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]

    def _try_load_pred_from_disk(self, k, sample_name: str, cache_root: str = "runs/cache") -> Optional[np.ndarray]:
        """
        Try to read preds/<sample_name>/<stem>.npy under cache_root.
        """
        if self.path_for_cache is not None:
            cache_root = self.path_for_cache
        try:
            stem = self.model_source_name(k)
            fp = Path(cache_root) / "preds" / sample_name / f"{stem}.npy"
            if fp.exists():
                return np.load(fp)
        except Exception:
            pass
        return None
    
    def _try_seed_raw_preds(self, keys: list, sample_name: str, cache_root : str = 'runs/cache') -> None:
        """
        RAM -> (optional) user-provided cache duck-type -> disk paths. No compute.
        """
        if getattr(self, "raw_predictions", None) is None:
            self.raw_predictions = {}
        if self.path_for_cache is not None:
            cache_root = self.path_for_cache 
        for k in keys:
            if (k, sample_name) in self.raw_predictions:
                continue
            arr = self._try_load_pred_from_disk(k, sample_name, cache_root=cache_root)
            if arr is not None:
                self.raw_predictions[(k, sample_name)] = arr
            
    def _try_get_entropy(self, keys: list, sample_name: str, cache_root: str = "runs/cache") -> Optional[np.ndarray]:
        """
        RAM (local) -> user-provided cache -> disk path. No compute.
        """
        if self.path_for_cache is not None:
            cache_root = self.path_for_cache
        if not hasattr(self, "_entropy_cache"):
            self._entropy_cache = {}  # (sig, sample_name) -> np.ndarray
        sig = self._committee_sig_from_keys(keys)
        if sig is None:
            return None

        # RAM
        if (sig, sample_name) in self._entropy_cache:
            return self._entropy_cache[(sig, sample_name)]


        # Disk: metrics/<sample_name>/entropy__<sig>.npy
        try:
            fp = Path(cache_root) / "metrics" / sample_name / f"entropy__{sig}.npy"
            if fp.exists():
                ent = np.load(fp)
                self._entropy_cache[(sig, sample_name)] = ent
                return ent
        except Exception:
            pass

        return None
    def cross_entropy(self, arrays: Sequence[np.ndarray]) -> float:
        """
        Compute  f = - sum_{k != l} sum_{i=1}^m a_{i,k} * log( a_{i,l} )
        where each input is an array of shape (m, 1) (or (m,)).

        Parameters
        ----------
        arrays : sequence of np.ndarray
            N arrays, each of shape (m,1) or (m,). They should all have the same m.

        Returns
        -------
        float
            The scalar value of the cross entropy f among the arrays

        Notes
        -----
        - Time complexity is O(m N^2) dominated by the matrix multiply, where N is the length of arrays.
        """
        if len(arrays) == 0:
            return 0.0

        # Stack to shape (m, N)
        cols = [np.asarray(a).reshape(-1) for a in arrays]
        m = cols[0].shape[0]
        if any(c.shape[0] != m for c in cols):
            raise ValueError("All arrays must have the same first dimension m.")
        A = np.column_stack(cols)  # shape (m, N)

        log_base = self.target  # uses your class' target as log base
        L = np.log(A + 1e-12) / np.log(log_base)

        M = A.T @ L  # shape (N, N), M[k, l] = sum_i a_{i,k} * log(a_{i,l})
        total = -(np.sum(M) - np.trace(M))  # sum over k != l, then negate
        return float(total)

    def get_entropy_array(
        self,
        keys: List[Tuple],
        sample_name: str,
        cache_root: str = "./data/cache",
        sample: Optional[np.ndarray] = None,
        preds: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        sample : np.ndarray
            Input batch X with shape (n, ...). Provide this unless you supply `preds`.
        preds : dict[str, np.ndarray]
            Optional precomputed raw predictions per key; if provided they will be cached.

        Returns
        -------
        np.ndarray
            Entropy values of shape (n,)
        """
        if self.path_for_cache is not None:
            cache_root = self.path_for_cache
        # try to get from RAM or disk cache
        ent = self._try_get_entropy(keys, sample_name, cache_root=cache_root)
        if ent is not None:
            return ent
        
        if sample is None and preds is None:
            raise ValueError("Either sample or preds must be provided")

        # initialize raw-pred cache
        if getattr(self, "raw_predictions", None) is None:
            self.raw_predictions = {}

        if preds is not None:
            for k, v in preds.items():
                self.raw_predictions[(k, sample_name)] = v
        
        self._try_seed_raw_preds(keys, sample_name, cache_root=cache_root)
        have_all = all((k, sample_name) in self.raw_predictions for k in keys)
        
        # ensure predictions exist for all keys
        if sample is not None and not have_all:
            for k in keys:
                if (k, sample_name) not in self.raw_predictions:
                    self.raw_predictions[(k, sample_name)] = self.getModel(k).predict(sample, verbose=0)

        # stacked shape: (n, K, C)  — n samples, K models, C classes
        stacked = np.stack([self.raw_predictions[(k, sample_name)] for k in keys], axis=1)
        n = stacked.shape[0]

        # compute cross-entropy across model distributions per sample
        ent = np.zeros(n, dtype=np.float64)
        for i in range(n):
            # arrays is a sequence of length K; each item is (C,)
            ent[i] = self.cross_entropy(stacked[i, :])

        return ent

    def predict(
        self,
        x,
        x_name: str,
        # load cache files if available; you must provide
        cache_root: str = "runs/cache",
        # committees you already have:
        committee_Judge=None,     # list of model keys for low-entropy bucket
        committee_experts=None,   # list of model keys for high-entropy bucket
        entropy_threshold=0.5,
        # keep parent signature defaults for compatibility:
        wantAcc=None, maskInWantAcc=None, keys=None,
        batch_size: int = 128, force_batch: bool = False,
        active: bool = True, verbose: int = 0,
        voteBy='weighAcc', onlyFineCanVote: bool = False,
        # targetTree=None, keysTree=None, node=(),
        fullAns=None, certPart=None, originalAns=None,
        pred=None, pred_useHist=None, certs=None,
        out=None, modelAcc=None, y=None, reportSeq=None,
        predOutputFile: str = None, evalOutputFile: str = None,
        show_av_acc: bool = False, show_simple_vote: bool = False,
        count=0, useHistory=None, write_story: bool = True,
        display_: bool = True,
    ):
        """
        Routes inputs by entropy to two committees and predicts on a flat, full target only.
        """
        if committee_Judge is None or committee_experts is None:
            raise ValueError("Please provide committee_Judge and committee_experts (lists of model keys).")
        keys_all = list(committee_Judge) + list(committee_experts)
        if self.path_for_cache is not None:
            cache_root = self.path_for_cache
        # try to load raw preds from disk cache
        self._try_seed_raw_preds(
            keys=keys_all,
            sample_name=x_name,
            cache_root=cache_root,
        )
        # basic safety / defaults
        assert voteBy in ('weighAcc', 'order')
        if useHistory is not None and evalOutputFile is not None:
            assert useHistory != evalOutputFile

        if committee_Judge is None or committee_experts is None:
            raise ValueError("Please provide committee_Judge and committee_experts (lists of model keys).")

        keys = list(committee_Judge) + list(committee_experts)

        if pred is None: pred = []
        if certs is None: certs = []
        if out is None: out = []
        if modelAcc is None: modelAcc = {}
        if originalAns is None: originalAns = {}
        if wantAcc is None:
            wantAcc = [0.5, 0.7310585786300049, 0.8807970779778823, 0.9525741268224334,
                       0.9820137900379085, 0.9933071490757153, 0.9975273768433653,
                       0.9990889488055994, 0.9996646498695336, 0.9998766054240137]
        assert isinstance(wantAcc, (list, np.ndarray, float))
        if isinstance(wantAcc, float):
            wantAcc = [wantAcc]
        wantAcc = np.array(wantAcc, dtype=float)
        if wantAcc[0] != 0.0:
            wantAcc = np.insert(wantAcc, 0, 0.0)

        # flat target tree (root only); no recursion/children
        flat_targetTree = {(): tuple(range(self.target))}
        flat_keysTree_low  = {(): list(committee_Judge)}
        flat_keysTree_high = {(): list(committee_experts)}

        # compute entropies & masks
        entropy_array = self.get_entropy_array(keys, sample_name=x_name, sample=x)
        if entropy_array.shape[0] != len(x):
            raise ValueError("get_entropy_array must return shape (len(x),).")
        mask_low  = (entropy_array < entropy_threshold)
        mask_high = ~mask_low

        def _expand_mask_per_wantAcc(vec: np.ndarray) -> List[np.ndarray]:
            # parent expects maskInWantAcc as a list (len=|wantAcc|) of boolean arrays
            return [vec.copy() for _ in range(len(wantAcc))]

        mask_low_list  = _expand_mask_per_wantAcc(mask_low)
        mask_high_list = _expand_mask_per_wantAcc(mask_high)

        # pass 1: low-entropy bucket
        fullAns, certPart, pred, certs, out, originalAns, modelAcc, pred_useHist = super().predict(
            x, x_name,
            wantAcc=wantAcc,
            maskInWantAcc=mask_low_list,
            keys=list(committee_Judge),
            batch_size=batch_size,
            force_batch=force_batch,
            active=False,               # unchanged
            verbose=verbose,
            voteBy=voteBy,
            onlyFineCanVote=onlyFineCanVote,
            targetTree=flat_targetTree,
            keysTree=flat_keysTree_low,
            node=(),
            fullAns=fullAns,
            certPart=certPart,
            originalAns=originalAns,
            pred=pred,
            pred_useHist=pred_useHist,
            certs=certs,
            out=out,
            modelAcc=modelAcc,
            y=y,
            reportSeq=reportSeq,
            predOutputFile=predOutputFile,
            evalOutputFile=evalOutputFile,
            show_av_acc=show_av_acc,
            show_simple_vote=show_simple_vote,
            count=count,
            useHistory=useHistory,
            write_story=False,          # write once after both passes
            display_=display_,
        )

        # pass 2: high-entropy bucket
        fullAns, certPart, pred, certs, out, originalAns, modelAcc, pred_useHist = super().predict(
            x, x_name,
            wantAcc=wantAcc,
            maskInWantAcc=mask_high_list,
            keys=list(committee_experts),
            batch_size=batch_size,
            force_batch=force_batch,
            active=False,
            verbose=verbose,
            voteBy=voteBy,
            onlyFineCanVote=onlyFineCanVote,
            targetTree=flat_targetTree,
            keysTree=flat_keysTree_high,
            node=(),
            fullAns=fullAns,
            certPart=certPart,
            originalAns=originalAns,
            pred=pred,
            pred_useHist=pred_useHist,
            certs=certs,
            out=out,
            modelAcc=modelAcc,
            y=y,
            reportSeq=reportSeq,
            predOutputFile=predOutputFile,
            evalOutputFile=evalOutputFile,
            show_av_acc=show_av_acc,
            show_simple_vote=show_simple_vote,
            count=count,
            useHistory=useHistory,
            write_story=write_story,    # write after both passes
            display_=display_,
        )

        return fullAns, certPart, pred, certs, out, originalAns, modelAcc, pred_useHist