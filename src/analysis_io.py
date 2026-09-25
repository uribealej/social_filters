"""Small file-discovery and interactive inspection utilities."""

from pathlib import Path

import numpy as np

def find_file_with_suffix(folder: Path, suffix: str) -> Path:
    matches = list(folder.glob(f"*{suffix}"))
    if len(matches) == 0:
        raise FileNotFoundError(f"No file ending with '{suffix}' in {folder}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files ending with '{suffix}' in {folder}: {matches}")
    return matches[0]

def inspect_obj(obj, name="obj", max_items=5, level=0):
    """
    Quick inspection of a Python object:
    - prints type
    - for numpy arrays: shape + dtype
    - for dicts: shows keys and value types (and shapes if arrays)
    - for lists/tuples: length and types of first items
    """
    indent = "  " * level
    t = type(obj)
    print(f"{indent}{name}: {t}")

    # --- numpy array ---
    if isinstance(obj, np.ndarray):
        print(f"{indent}  shape={obj.shape}, dtype={obj.dtype}")
        return

    # --- dict ---
    if isinstance(obj, dict):
        print(f"{indent}  dict with {len(obj)} keys:")
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                print(f"{indent}    ... ({len(obj) - max_items} more keys)")
                break
            line = f"{indent}    key={repr(k)} -> type={type(v)}"
            if isinstance(v, np.ndarray):
                line += f", shape={v.shape}, dtype={v.dtype}"
            elif isinstance(v, (list, tuple)):
                line += f", len={len(v)}"
            print(line)
        return

    # --- list / tuple ---
    if isinstance(obj, (list, tuple)):
        print(f"{indent}  {t.__name__} of length {len(obj)}")
        for i, item in enumerate(obj[:max_items]):
            line = f"{indent}    [{i}]: type={type(item)}"
            if isinstance(item, np.ndarray):
                line += f", shape={item.shape}, dtype={item.dtype}"
            elif isinstance(item, (list, tuple)):
                line += f", len={len(item)}"
            print(line)
        if len(obj) > max_items:
            print(f"{indent}    ... ({len(obj) - max_items} more items)")
        return

    # --- fallback: just print value ---
    try:
        print(f"{indent}  value={repr(obj)}")
    except Exception:
        pass
