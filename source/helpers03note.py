from pathlib import Path
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import AutoProjectedGradientDescent
import numpy as np
from .utils import collect_model_paths
from .utils import load_prob_models
import tensorflow as tf
from tensorflow import keras
import glob
from typing import List



class ProbAverageEnsemble(keras.Model):
    def __init__(self, prob_models: List[keras.Model], eps: float = 1e-12):
        super().__init__()
        self.prob_models = prob_models
        self.eps = eps

    def call(self, x, training=False):
        ps = [m(x, training=training) for m in self.prob_models]  # probs
        p = tf.add_n(ps) / float(len(ps))
        return tf.clip_by_value(p, self.eps, 1.0)
def build_art_classifier(prob_models: List[keras.Model]) -> TensorFlowV2Classifier:
    ens = ProbAverageEnsemble(prob_models)
    clf = TensorFlowV2Classifier(
        model=ens,
        nb_classes=10,
        input_shape=(32, 32, 3),
        clip_values=(0.0, 1.0),
        loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    )
    return clf

def save_npz(path: Path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **kwargs)
    print("Saved:", path.name)

def run_apgdce_and_save(
    clf: TensorFlowV2Classifier,
    X: np.ndarray,
    Y: np.ndarray,
    split : str,
    out_path: Path,
    eps: float,
    eps_step: float,
    max_iter: int,
    nb_random_init: int,
    batch_size: int = 64,
    tag: str = ""
    
):
    attack = AutoProjectedGradientDescent(
        estimator=clf,
        norm=2,
        eps=float(eps),
        eps_step=float(eps_step),
        max_iter=int(max_iter),
        nb_random_init=int(nb_random_init),
        targeted=False,
        batch_size=int(batch_size),
        loss_type="cross_entropy",
        verbose=False,
    )
    x_adv = attack.generate(x=X, y=None).astype(np.float32)

    save_npz(
        out_path,
        x_adv=x_adv,
        y=Y.astype(np.int64),
        split = split,
        norm="L2",
        eps=np.float32(eps),
        eps_step=np.float32(eps_step),
        max_iter=np.int32(max_iter),
        nb_random_init=np.int32(nb_random_init),
        tag = tag
    )
    return x_adv

def gen_for_ensemble(tag: str, model_paths: List[str], APGD_SETTINGS, splits, APGD_OUT: Path):
    if len(model_paths) == 0:
        print(f"[{tag}] no models found -> skipping")
        return

    prob_models = load_prob_models(model_paths)
    clf = build_art_classifier(prob_models)

    for eps, eps_step, max_iter, rinit in APGD_SETTINGS:
        eps_tag = str(eps).replace(".", "p")
        step_tag = str(eps_step).replace(".", "p")

        for split_name, Xs, Ys in splits:
            out_name = f"{split_name}_apgdce_l2_eps{eps_tag}_step{step_tag}_it{max_iter}_rinit{rinit}_against_{tag}.npz"
            out_path = APGD_OUT / out_name

            if out_path.exists():
                continue  # avoid re-generation

            _ = run_apgdce_and_save(
                clf,
                Xs.astype(np.float32),
                Ys.astype(np.int64),
                split = split_name,
                out_path=out_path,
                eps=eps,
                eps_step=eps_step,
                max_iter=max_iter,
                nb_random_init=rinit,
                batch_size=64,
                tag = tag
            )

def merge_train_chunks(tag: str, APGD_SETTINGS, APGD_OUT: Path):
    eps, eps_step, max_iter, rinit = APGD_SETTINGS
    eps_tag = str(eps).replace(".", "p")
    step_tag = str(eps_step).replace(".", "p")

    pattern = str(APGD_OUT / f"train_all_*_apgdce_l2_eps{eps_tag}_step{step_tag}_it{max_iter}_rinit{rinit}_against_{tag}.npz")
    paths = sorted(glob.glob(pattern))
    if len(paths) == 0:
        print("No chunk files:", pattern)
        return

    x_list, y_list = [], []
    for p in paths:
        d = np.load(p)
        x_list.append(d["x_adv"])
        y_list.append(d["y"])

    x_adv = np.concatenate(x_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)

    out_name = f"train_all_apgdce_l2_eps{eps_tag}_step{step_tag}_it{max_iter}_rinit{rinit}_against_{tag}.npz"
    out_path = APGD_OUT / out_name
    save_npz(
        out_path,
        x_adv=x_adv,
        y=y,
        split = 'train',
        norm="L2",
        eps=np.float32(eps),
        eps_step=np.float32(eps_step),
        max_iter=np.int32(max_iter),
        nb_random_init=np.int32(rinit),
        tag = tag
    )

