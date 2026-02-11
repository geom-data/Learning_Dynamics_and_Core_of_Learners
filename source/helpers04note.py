
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from typing import Dict
import numpy as np
from . import utils
from . import cache_store

def load_models_from_dir(model_dir: Path, pattern: str="*.keras", exclude_substr: str="original") -> Dict[str, tf.keras.Model]:
    md = {}
    for p in sorted(model_dir.glob(pattern)):
        if exclude_substr and exclude_substr in p.name:
            continue
        md[p.name] = load_model(str(p))
    return md


    
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
            outdir = self.caching.root / "preds" / sample_name
            fp = outdir / f"{self.caching._stem(model_name)}.npy"
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
                
                pred = model.predict(x, verbose=0)
                self.caching.set_pred(model_name, sample_name, pred)
#                     print('Done.')
                    
    def caching_ent(self, sample_name):
        # Get predictions for each model.
        sig = self.caching._committee_sig(list(self.prob_models.keys()))
        outdir = self.caching.root / "metrics" / sample_name
        fp = outdir / f"entropy__{sig}.npy"
        # Caching entropy
        if fp and Path(fp).exists():
#             print('sample', sample_name, 'entropy cached already')
            # Already cached and file exists: fine.
            pass

        else:
            # No index entry; check if entropy file exists under naming convention.
            
            arrays = np.array([self.caching.get_pred(model_name, sample_name) for model_name in sorted(self.prob_models.keys())])

            ent = utils.cross_entropy(arrays, log_base = 10)
            self.caching.set_entropy(sorted(self.prob_models.keys()), sample_name, ent)
    