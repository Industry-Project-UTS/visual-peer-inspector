"""Selected-view, non-pickle loading for compact current cohort caches."""
from pathlib import Path
import numpy as np
from vpi_object_bank.storage import digest_file, read_json
from vpi_remaining_evaluation.cohort import selected_inputs, _reject_labels


def current_inputs(payload, views, mode, cache_root):
    """Accept legacy inline caches or exact sealed NPZ references, never both.

    Callers verify the complete cache seal once before this function. Each
    selected tensor is additionally checked against that seal before loading.
    Unselected views are not loaded or passed to the scorer.
    """
    _reject_labels(payload)
    observations = payload['bank_observations']
    if not any('patches_file' in row for row in observations):
        return selected_inputs(payload, views, mode)
    if any(set(row) != {'view_id', 'candidate_id', 'object_id', 'patches_file', 'patches_key'} for row in observations):
        raise ValueError('Inline and external bank tensors cannot be mixed')
    root = Path(cache_root).resolve()
    lock = read_json(root / 'artifact_lock.json')
    selected = [row for row in observations if row['view_id'] in views]
    loaded = {}
    for relative in sorted({row['patches_file'] for row in selected}):
        path = root / relative
        if not isinstance(relative, str) or not relative.startswith('tensors/') or path.suffix != '.npz' or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Unsafe bank tensor reference')
        if relative not in lock['files'] or digest_file(path) != lock['files'][relative]:
            raise ValueError('Bank tensor is missing, corrupt, or outside the cache seal')
        owners = [row for row in selected if row['patches_file'] == relative]
        keys = [row['patches_key'] for row in owners]
        if len(keys) != len(set(keys)) or len({row['view_id'] for row in owners}) != 1:
            raise ValueError('Bank tensor aliases candidates or crosses views')
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(keys):
                raise ValueError('Bank tensor file has missing or unexpected arrays')
            for key in keys:
                matrix = arrays[key]
                if matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[1] != 384 or not np.isfinite(matrix).all():
                    raise ValueError('Bank tensors require finite FP32 DINO matrices')
                loaded[relative, key] = matrix.copy()
    concrete = [{key: row[key] for key in ('view_id', 'candidate_id', 'object_id')} |
        {'patches': loaded[row['patches_file'], row['patches_key']]} for row in selected]
    return selected_inputs({**payload, 'bank_observations': concrete}, views, mode)
