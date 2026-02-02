from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union
import json

import pandas as pd

from .. import get_logger, paths
from ..config import CacheConfig


logger = get_logger(__name__)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return md5(raw).hexdigest()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _csv_read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _csv_write(path: Path, df: pd.DataFrame) -> None:
    _ensure_dir(path.parent)
    df.to_csv(path)


def _json_read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_write(path: Path, data: Mapping[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(dict(data), indent=2, sort_keys=True, default=str), encoding="utf-8")


@dataclass(frozen=True)
class CachePaths:
    base_dir: Path

    def subdir(self, name: str) -> Path:
        p = self.base_dir / name
        _ensure_dir(p)
        return p

    def csv_path(self, subdir: str, payload: Mapping[str, Any]) -> Path:
        key = _hash_payload(payload)
        return self.subdir(subdir) / f"{key}.csv"

    def json_path(self, subdir: str, payload: Mapping[str, Any]) -> Path:
        key = _hash_payload(payload)
        return self.subdir(subdir) / f"{key}.json"

    def meta_path(self, subdir: str, payload: Mapping[str, Any]) -> Path:
        key = _hash_payload(payload)
        return self.subdir(subdir) / f"{key}.meta.json"


def cache_paths(cache: Optional[CacheConfig] = None) -> CachePaths:
    cfg = cache or CacheConfig(enabled=True, dir=paths.cache)
    return CachePaths(base_dir=Path(cfg.dir))


def read_frame(
    subdir: str,
    payload: Mapping[str, Any],
    *,
    cache: Optional[CacheConfig] = None,
) -> Optional[pd.DataFrame]:
    cfg = cache or CacheConfig(enabled=True, dir=paths.cache)
    if not cfg.enabled:
        return None
    cp = cache_paths(cfg)
    path = cp.csv_path(subdir, payload)
    if not path.exists():
        return None
    try:
        return _csv_read(path)
    except Exception:
        return None


def write_frame(
    subdir: str,
    payload: Mapping[str, Any],
    df: pd.DataFrame,
    *,
    cache: Optional[CacheConfig] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> None:
    cfg = cache or CacheConfig(enabled=True, dir=paths.cache)
    if not cfg.enabled:
        return
    cp = cache_paths(cfg)
    path = cp.csv_path(subdir, payload)
    _csv_write(path, df)
    meta_payload: Dict[str, Any] = {
        "written_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "subdir": subdir,
        "payload": dict(payload),
    }
    if meta:
        meta_payload.update(dict(meta))
    _json_write(cp.meta_path(subdir, payload), meta_payload)


def read_series(
    subdir: str,
    payload: Mapping[str, Any],
    *,
    name: str = "value",
    cache: Optional[CacheConfig] = None,
) -> Optional[pd.Series]:
    df = read_frame(subdir, payload, cache=cache)
    if df is None or len(df) == 0:
        return None
    if df.shape[1] == 1:
        s = df.iloc[:, 0]
        s.name = name
        return s
    if name in df.columns:
        s = df[name]
        s.name = name
        return s
    s = df.iloc[:, 0]
    s.name = name
    return s


def write_series(
    subdir: str,
    payload: Mapping[str, Any],
    series: pd.Series,
    *,
    cache: Optional[CacheConfig] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> None:
    df = series.to_frame(name=series.name or "value")
    write_frame(subdir, payload, df, cache=cache, meta=meta)


def read_json(
    subdir: str,
    payload: Mapping[str, Any],
    *,
    cache: Optional[CacheConfig] = None,
) -> Optional[Dict[str, Any]]:
    cfg = cache or CacheConfig(enabled=True, dir=paths.cache)
    if not cfg.enabled:
        return None
    cp = cache_paths(cfg)
    path = cp.json_path(subdir, payload)
    if not path.exists():
        return None
    try:
        return _json_read(path)
    except Exception:
        return None


def write_json(
    subdir: str,
    payload: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    cache: Optional[CacheConfig] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> None:
    cfg = cache or CacheConfig(enabled=True, dir=paths.cache)
    if not cfg.enabled:
        return
    cp = cache_paths(cfg)
    _json_write(cp.json_path(subdir, payload), data)
    meta_payload: Dict[str, Any] = {
        "written_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "subdir": subdir,
        "payload": dict(payload),
    }
    if meta:
        meta_payload.update(dict(meta))
    _json_write(cp.meta_path(subdir, payload), meta_payload)
