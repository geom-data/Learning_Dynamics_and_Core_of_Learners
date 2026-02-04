import numpy as np
from pathlib import Path
import shutil

def entropy_threshold(certainty : float, base : int = 10):
    '''
    For a inferred distribution by single model over C classes, if desired certainty level is achieved, then upper bound for the information entropy is computed as follows:
    entropy_threshold = - certainty*log(certainty) / log(base) - (1 - certainty)*log( (1 - certainty) / (C - 1) ) / log(base)
    where C is the number of classes.
    
    This entropy threshold attains about 0.469 when certainty = 0.9 and C = 2. If certainty = 0.9 and C = 10, then the entropy threshold is about 0.237.
    '''
    if isinstance(base,int) is False or base < 2:
        raise ValueError("Base must be an integer greater than or equal to 2.")
    
    
    x = certainty
    C = base
    return -x * np.log(x) / np.log(C) - (1 - x) * np.log((1 - x) / (C - 1)) / np.log(C)


def cross_entropy(arrays) -> float:
        """
        Compute  f = - sum_{k,l} sum_{i=1}^m a_{i,k} * log( a_{i,l} )
        where each input is an array of shape (m, 1) (or (m,)).

        Parameters
        ----------
        arrays : sequence of np.ndarray
            N arrays, each of shape (m,1) or (m,). They should all have the same m.

        Returns
        -------
        float
            The scalar value of the cross entropy f among the arrays

        -----
        """
        n = len(arrays)
        if n == 0:
            raise ValueError("At least one array is required.")

        # Stack to shape (m, n)
        cols = [np.asarray(a).reshape(-1) for a in arrays]
        m = cols[0].shape[0]
        if any(c.shape[0] != m for c in cols):
            raise ValueError("All arrays must have the same first dimension m.")
        A = np.column_stack(cols)  # shape (m, n)

        log_base = m  
        L = np.log(A + 1e-12) / np.log(log_base)

        M = A.T @ L  # shape (n, n), M[k, l] = sum_i a_{i,k} * log(a_{i,l})
        total = -np.sum(M) # sum over all pair
        
        return float(total)/(n**2)
    
def clear_folder(folder: str | Path, *, keep_folder: bool = True, dry_run: bool = False) -> None:
    """
    Delete all contents inside folder.
    """
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

