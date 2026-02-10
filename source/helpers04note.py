
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from . import utils
from . import cache_store
def load_models_from_dir(model_dir: Path, pattern: str="*.keras", exclude_substr: str="original") -> Dict[str, tf.keras.Model]:
    md = {}
    for p in sorted(model_dir.glob(pattern)):
        if exclude_substr and exclude_substr in p.name:
            continue
        md[p.name] = load_model(str(p))
    return md


def plot_ent_th_sweep(
    df : pd.DataFrame,
    overall_acc,
    sample_name="AA_RBST_test",
    outdir=None,
    vline_ent_ths: list[float] = [0.0414, 1.0],
    show_legend: bool = False,
    legend_outside: bool = True,

):
    ''' vline_ent_ths = [ent_th found on validation sample, high entropy threshold (arbitrary)]'''
    assert len(vline_ent_ths) == 2, "vline_ent_ths must have two thresholds (low and high)."

    df = df.copy()
    df["ent_th"] = df["ent_th"].astype(float)
    df = df.sort_values("ent_th").reset_index(drop=True)

    x = df["ent_th"].to_numpy()
    core_cnt = df["core count"].to_numpy(dtype=float)
    high_cnt = df["high count"].to_numpy(dtype=float)
    acc1 = df["acc1"].to_numpy(dtype=float)
    acc2 = df["acc2"].to_numpy(dtype=float)

    total_n = core_cnt + high_cnt
    N = float(total_n[0])

    def _save(fig, name):
        if outdir is None:
            return
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, name), dpi=300)

    def nearest_idx(arr, val):
        arr = np.asarray(arr, dtype=float)
        return int(np.argmin(np.abs(arr - float(val))))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(x, core_cnt / N, label="core coverage", linestyle="--", color="orange")
    ax1.plot(x, high_cnt / N, label="high coverage", linestyle="-.", color="blue")

    ax2.plot(x, acc1, linestyle="-", label="acc (core)", color="black")
    ax2.plot(x, acc2, linestyle="-", label="acc (high)", color="red")

    ax2.axhline(float(overall_acc), linestyle=":")
    ax2.annotate(
        f"overall acc\n{overall_acc:.4f}",
        (x[-1], float(overall_acc)),
        textcoords="offset points",
        xytext=(18, 8),
        ha="left",
        va="bottom",
        fontsize=9,
    )

    ax1.set_xlabel("Entropy Threshold")
    ax1.set_ylabel("Coverage")
    ax2.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.1)
    ax2.set_ylim(0, 1.1)

    ax1.grid(False)
    ax2.grid(False)
    plt.yticks([])

    core_loc = nearest_idx(x, vline_ent_ths[0])

    ax1.axvline(x[core_loc], linestyle="--", alpha=0.7)
    ax1.annotate(
        f"{vline_ent_ths[0]:.1f}",
        (x[core_loc], 0),
        textcoords="offset points",
        xytext=(6, -6),
        ha="left",
        va="top",
        fontsize=11,
    )

    ax1.plot([x[core_loc]], [core_cnt[core_loc] / N], marker="o", linestyle="None", color="black")
    ax1.annotate(
        f"{core_cnt[core_loc] / N:.4f}",
        (x[core_loc], core_cnt[core_loc] / N),
        textcoords="offset points",
        xytext=(-6, 4),
        ha="right",
        va="bottom",
        fontsize=11,
    )

    ax1.set_xscale("log")

    fig.subplots_adjust(right=0.85, top=0.82)

    if show_legend:
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2

        if legend_outside:
            ax1.legend(
                lines,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=len(labels),
                frameon=True,
                borderaxespad=0.0,
                handlelength=2.0,
                columnspacing=1.2,
            )
        else:
            ax1.legend(lines, labels, loc="best")

    _save(fig, f"{sample_name}_counts_and_acc_dualaxis.png")
    plt.show()
    
class ProbAverageEnsembleFromProbModels(tf.keras.Model):
    def __init__(self, prob_models : dict, eps=1e-12, cache_root : str = './data/cache/'):
        super().__init__()
        self.prob_models = prob_models
        self.eps = eps
        self.caching = cache_store.ResultStore(root = cache_root)
        
    def probs(self, x, sample_name : str = ''):
        '''
        return average probability (average softmax value)
        '''
        if sample_name:
            self.caching_pred(x, sample_name)
            preds = [self.caching.get_pred(model_name, sample_name) for model_name in sorted(self.prob_models)]
        else:
            preds = [self.prob_models[k].predict(x,verbose = 0, batch_size = 64) for k in sorted(self.prob_models)]
        
        return np.mean(preds, axis=0)
    def core_preds(self, x, sample_name : str = '', ent_th : float = 0.1):
        '''
        return average probability (average softmax value)
        '''
        if sample_name:
            self.caching_pred(x, sample_name)
            preds = [self.caching.get_pred(model_name, sample_name) for model_name in sorted(self.prob_models)]
        else:
            preds = [self.prob_models[k].predict(x,verbose = 0, batch_size = 64) for k in sorted(self.prob_models)]
        preds = np.array(preds)
        
        if sample_name:
            self.caching_ent(sample_name)
            ent = self.caching.get_entropy(sorted(self.prob_models), sample_name)
        else:
            ent = []
            for i in range(preds.shape[1]):
                ent.append(self.cross_entropy(preds[:,i,: ]))
            ent = np.array(ent)
        
        core = ent<ent_th
        
        preds = preds[:, core, :]
        core_preds = np.mean(preds, axis=0)
        
        return core_preds, core, ent
        
            
    
    
    def caching_pred(self, x, sample_name):
        for model_name, model in self.prob_models.items():
            key = (self.caching.int_from_model_path(model_name), sample_name)
            fp = self.caching._preds_idx.get(key)

            if fp and Path(fp).exists():
                # Already cached and file exists: nothing to do.
#                 print(f'{model_name} already made predictions on {sample_name} and it was cached.')
                
                continue

            elif fp and not Path(fp).exists():
                # Index says there is a file, but it doesn't exist anymore.
                # Recompute and overwrite.
#                 print(f"Index says there is a file, but it doesn't exist anymore. {model_name}  on {sample_name}. Recompute and Overwrite.")

                pred = model.predict(x, verbose=0)
                self.caching.set_pred(model_name, sample_name, pred)

            else:
                # No index entry; check if file exists under the naming convention.
#                 print(f'{model_name} make predictions on {sample_name} and it is cached.')
                outdir = self.caching.root / "preds" / sample_name
                fp = outdir / f"{self.caching._stem(model_name)}.npy"
                if Path(fp).exists():
#                     print('File exists but is not in the index: add it.')
                    # File exists but is not in the index: add it.
                    pred = np.load(fp)
                    key = (self.caching.int_from_model_path(model_name), sample_name)
                    self.caching._preds_idx[key] = str(fp)
                    self.caching._dump_idx(self.caching._preds_idx_path, self.caching._preds_idx)
                else:
#                     print('processing..', end = ' ')
                    # Compute and cache.
                    pred = model.predict(x, verbose=0)
                    self.caching.set_pred(model_name, sample_name, pred)
#                     print('Done.')
                    
    def caching_ent(self, sample_name):
        # Get predictions for each model.
        sig = self.caching._committee_sig(list(self.prob_models.keys()))
        key = ("entropy", sig, sample_name)
        fp = self.caching._metrics_idx.get(key)

        # Caching entropy
        if fp and Path(fp).exists():
#             print('sample', sample_name, 'entropy cached already')
            # Already cached and file exists: fine.
            pass

        elif fp and not Path(fp).exists():
            # Index points to missing file: recompute and overwrite.
            arrays = np.array([self.caching.get_pred(model_name, sample_name) for model_name in sorted(self.prob_models.keys())])

            ent = []
            for i in range(arrays.shape[1]):
                ent.append(self.cross_entropy(arrays[:,i,: ]))
            ent = np.array(ent)
            self.caching.set_entropy(sorted(self.prob_models.keys()), sample_name, ent)

        else:
            # No index entry; check if entropy file exists under naming convention.
            outdir = self.caching.root / "metrics" / sample_name
            fp = outdir / f"entropy__{sig}.npy"
            if Path(fp).exists():
                key = ("entropy", sig, sample_name)
                self.caching._metrics_idx[key] = str(fp)
                self.caching._dump_idx(self.caching._metrics_idx_path, self.caching._metrics_idx)
            else:
                # Compute and cache.
                arrays = np.array([self.caching.get_pred(model_name, sample_name) for model_name in sorted(self.prob_models.keys())])

                ent = utils.cross_entropy(arrays, log_base = 10)
                self.caching.set_entropy(sorted(self.prob_models.keys()), sample_name, ent)
    