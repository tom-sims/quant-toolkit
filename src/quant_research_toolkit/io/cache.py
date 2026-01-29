import hashlib
import json
from pathlib import Path

import pandas as pd


def _stable_json(obj):
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def _hash_key(parts):
    raw = _stable_json(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class DiskCache:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key):
        return self.cache_dir / (key + ".parquet")

    def get(self, key):
        p = self._path(key)
        if not p.exists():
            return None
        return pd.read_parquet(p)

    def set(self, key, value):
        p = self._path(key)
        value.to_parquet(p, index=True)


def cached_dataframe(cache):
    """
    Decorator for caching functions that return a pandas DataFrame.
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            key = _hash_key({
                "fn": fn.__module__ + "." + fn.__name__,
                "args": args,
                "kwargs": kwargs,
            })
            hit = cache.get(key)
            if hit is not None:
                return hit

            out = fn(*args, **kwargs)
            if not isinstance(out, pd.DataFrame):
                raise TypeError(fn.__name__ + " must return a pandas DataFrame for caching.")
            cache.set(key, out)
            return out

        return wrapper
    return decorator
